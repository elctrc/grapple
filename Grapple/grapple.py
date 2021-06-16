import csv
import datetime
from itertools import product
import logging
import pprint
import re
import string
import yaml
from zipfile import ZipFile
import time
from haystack import Finder
from haystack.indexing.cleaning import clean_wiki_text
# from haystack.indexing.io import write_documents_to_db, fetch_archive_from_http
from haystack.indexing.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.database.memory import InMemoryDocumentStore

# Create logger object
logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s")


class HayLoader:
    """
    Generate Haystack model from config and run through a series of tests to assess performance

    Requirements:
    Haystack must be installed.
    """
    def __init__(self, config='config.yaml', test_set='test_set.yaml'):
        """Initialize and store config data from file containing levers"""
        # Load configuration and test set
        self.config = self.load_yaml(config)
        self.test_set = self.load_yaml(test_set)
        # Set our defaults
        self.document_store = InMemoryDocumentStore()
        # Set main configuration options
        self.source = self.config['source_dir']
        self.destination = self.config['destination_dir']
        self.store_type = self.config['store_type']
        self.pull_method = self.config['pull_method']
        self.unzip = self.config['unzip']
        self.use_gpu = self.config['use_gpu']
        self.print_results = self.config['print_results']
        # Elasticsearch options
        self.host = self.config['elasticsearch_options']['host']
        self.username = self.config['elasticsearch_options']['username']
        self.password = self.config['elasticsearch_options']['password']
        self.index = self.config['elasticsearch_options']['index']
        # Initialize Levers
        self.lever_sets = []
        # Initialize tracker for monitoring total permutations
        self.total_permutations = 0
        # Initialize list of dicts that will hold the final results
        self.all_models_and_levers = []
        logging.info('Hay loaded and ready to bale...')

    @staticmethod
    def load_yaml(filename):
        """
        Load yaml data from a file and store as an object
        """
        stream = open(filename, 'r')
        return yaml.load(stream, Loader=yaml.FullLoader)

    def get_levers(self):
        """
        Loop through each possible combination of levers and assemble lever packages
        into a single object
        """
        framework = self.config['framework']
        for framework_name in framework:
            for rm in framework[framework_name]['retrieve_method']:
                framework_levers = framework[framework_name]['retrieve_method'][rm]['levers']
                lever_permutations = self.make_permutations(framework_levers)
                self.total_permutations += len(lever_permutations)
                # Store all lever sets for access later
                self.lever_sets.append({
                    "levers": lever_permutations,
                    "retrieve_method": rm,
                    "reader_method": framework_name,
                    "f1_total_score": None,
                    "exact_match_total_score": None
                    })
        logging.info(f"Total Lever Permutations: {self.total_permutations}\n")
        time.sleep(5)

    @staticmethod
    def make_permutations(config):
        """
        Given a single config, expand out all possible permutations of the levers present
        """
        keys, values = zip(*config.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def setup_stack(self):
        """
        Setup Haystack model by providing config data to Haystack methods

        Attributes required:
            self.store_type
            self.pull_method
            self.source
            self.destination
        """
        if self.store_type == 'inmem':
            logging.info('Using default memory storage...')
        elif self.store_type == 'elastic':
            self.document_store = ElasticsearchDocumentStore(
                host=self.host,
                username=self.username,
                password=self.password,
                index=self.index)
        else:
            raise Exception('No proper document store specified in config.')

        if self.pull_method == 'gdrive':
            from google.colab import drive
            drive.mount('/content/drive')
            self.unzip_docs(folder_loc=self.source, dest=self.destination)
        elif self.pull_method == 's3':
            self.extract_from_s3(s3_url=self.source, doc_dir=self.destination)
        elif self.pull_method == 'local':
            if self.unzip:
                self.unzip_docs(folder_loc=self.source, dest=self.destination)
                logging.info('Local files unzipped...')
            else:
                logging.info('Using flat local files...')
        else:
            raise Exception('No proper pull method specified in config.')

        logging.info('Stack set...')

    @staticmethod
    def unzip_docs(folder_loc, dest):
        """
        Given directory with zip file, unzip and store in destination location
        """
        from zipfile import ZipFile

        with ZipFile(folder_loc, 'r') as zipped:
            zipped.extractall(dest)

    def extract_from_s3(self, s3_url, doc_dir):
        """Extract data from s3"""
        fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    def build_model(self, levers, retrieve_method, reader_method):
        """
        Build retriever, then load a local model and build a reader from either transformers
        or FARM framework. Then combine to form the finder
        """
        if retrieve_method == 'tfidf':
            from haystack.retriever.sparse import TfidfRetriever
            self.retriever = TfidfRetriever(document_store=self.document_store)
        elif retrieve_method == 'embeddings':
            from haystack.retriever.sparse import EmbeddingRetriever
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store,
                embedding_model=levers['embedding_model'],
                model_format=reader_method)
        else:
            raise Exception('No proper retrieve method specified in config.')

        if reader_method == 'farm':
            # Solve for yaml issue with Nonetype
            if levers['no_ans_boost'] == 'None':
                no_ans_boost = None
            else:
                no_ans_boost = levers['no_ans_boost']
            self.reader = FARMReader(
                model_name_or_path=levers['model'],
                context_window_size=levers['context_window_size'],
                use_gpu=self.use_gpu,
                no_ans_boost=no_ans_boost,
                top_k_per_candidate=levers['top_k_per_candidate'],
                top_k_per_sample=levers['top_k_per_sample'])
        elif reader_method == 'transformers':
            if self.use_gpu:
                gpu_ordinal = 0
            else:
                gpu_ordinal = -1
            self.reader = TransformersReader(
                model=levers['model'],
                # The tokenizer string is literally the name of the model ;)
                tokenizer=levers['model'],
                use_gpu=gpu_ordinal)
        else:
            raise Exception('No proper model specified in config.')

        self.finder = Finder(self.reader, self.retriever)

    @staticmethod
    def __average_list(list_of_numbers):
        """Returns the average of a list"""
        return sum(list_of_numbers) / len(list_of_numbers)

    def __average_response_time(self, time_list):
        """Returns the average response time for a given time-formatted list"""
        time_ints = []
        for time_string in time_list:
            time_ints.append(time_string.total_seconds())
        return self.__average_list(time_ints)

    def generate_answers(self, levers):
        """
        Given a set of questions and answers, this method runs through each, returning
        a predicted answer from Haystack Finder's get_answers method

        Each prediction is then scored using evaluate_answers, and the results
        are stored and then returned in the answer_set list of dictionaries
        """
        question_set = self.test_set['data']
        k_retrieve = levers['k_retrieve']
        k_read = levers['k_read']

        answer_set = []
        elapsed_times = []
        for qa in question_set:
            question = qa['question']
            ground_truth = qa['answer']

            begin_time = datetime.datetime.now()
            prediction = self.finder.get_answers(question=question, top_k_retriever=k_retrieve, top_k_reader=k_read)
            # print_answers(prediction, details="minimal")
            logging.info(f"Question: {question}\nPredicted Answer: {prediction}\n")
            end_time = datetime.datetime.now()
            total_elapsed = end_time - begin_time
            logging.info(f"Total elapsed time: {total_elapsed}.")
            elapsed_times.append(total_elapsed)
            predicted_answer = prediction['answers'][0]['answer']
            exact_match, f1, combined = self.evaluate_answers(predicted_answer, ground_truth)

            answer_set.append({
                "question": prediction['question'],
                "predicted_answer": predicted_answer,
                "context": prediction['answers'][0]['context'],
                # "offset_start": prediction['answers'][0]['offset_start'],
                # "offset_start_in_doc": prediction['answers'][0]['offset_start_in_doc'],
                # "offset_end": prediction['answers'][0]['offset_end'],
                # "offset_end_in_doc": prediction['answers'][0]['offset_end_in_doc'],
                "ground_truth": ground_truth,
                "exact_match_score": exact_match,
                "f1_score": f1,
                "combined_score": combined,
                "query_time": total_elapsed
            })

        time_ints = []
        for time_string in elapsed_times:
            time_ints.append(time_string.total_seconds())

        total_questions = len(elapsed_times)
        average_return = self.__average_list(time_ints)

        logging.info(f"You asked a total of {total_questions} questions with an average response time of {average_return}.")

        return(answer_set)

    @staticmethod
    def evaluate_answers(predicted_answer, ground_truth):
        """
        Match up answers and offsets to ground truth
        """
        pred_clean = HayLoader.normalize_answer(predicted_answer)
        truth_clean = HayLoader.normalize_answer(ground_truth)
        exact_match = 0
        f1 = HayLoader.calculate_f1(pred_clean, truth_clean)
        if pred_clean == truth_clean:
            exact_match = 1
        combined = f1 + exact_match
        return(exact_match, f1, combined)

    @staticmethod
    def calculate_f1(pred, truth):
        """
        Generate an f1 score from predictions
        """
        # Divide up pred and truth into vectors and count
        pred_words = pred.split(" ")
        pred_total_words = len(pred_words)
        truth_words = truth.split(" ")
        truth_total_words = len(truth_words)
        # precision = correct / total_pred_words
        total_correct = 0
        for word in pred_words:
            if word in truth_words:
                total_correct += 1
        precision = total_correct / pred_total_words
        # recall = correct / total_ground_words
        recall = total_correct / truth_total_words
        # f1 is the Harmonic mean of the recall and precision
        try:
            f1 = 2 / ((1 / precision) + (1 / recall))
        except ZeroDivisionError:
            f1 = 0

        return f1

    @staticmethod
    def normalize_answer(s):
        """
        Lower text and remove punctuation, articles and extra whitespace.
        Taken wholesale from SQuAD evaluations script:
        https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
        """
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def aggregate_scores(scores_dict):
        """Aggregate scores for one single pass"""
        em_mean = sum(score['exact_match_score'] for score in scores_dict) / len(scores_dict)
        f1_mean = sum(score['f1_score'] for score in scores_dict) / len(scores_dict)
        combined_mean = sum(score['combined_score'] for score in scores_dict) / len(scores_dict)

        return em_mean, f1_mean, combined_mean

    def run_models(self, levers, retrieve_method, reader_method):
        """
        Using the levers for the current permutation, setup and run the current model
        """
        self.setup_stack()
        # Funcion map to select correct clean function
        # Define additional clean functions as necessary
        function_map = {
            'clean_wiki_text': clean_wiki_text
        }

        dicts = convert_files_to_dicts(
            dir_path=self.destination,
            clean_func=function_map[levers['clean_function']],
            split_paragraphs=levers['split_paragraphs']
        )

        self.document_store.write_documents(dicts)
        # write_documents_to_db(
        #     document_store=self.document_store,
        #     document_dir=self.destination,
        #     clean_func=function_map[levers['clean_function']],
        #     only_empty_db=True,
        #     split_paragraphs=levers['split_paragraphs'])
        self.build_model(levers, retrieve_method, reader_method)
        return self.generate_answers(levers)

    @staticmethod
    def get_best_batch(lod, sort_key='combined'):
        """
        Find the best combination of levers, frameworks, and models

        sort_key: em, most_answers_scored, f1, combined
        """
        if sort_key == 'em':
            winning_index = max(range(len(lod)), key=lambda index: lod[index]['em_score'])
        elif sort_key == 'f1':
            winning_index = max(range(len(lod)), key=lambda index: lod[index]['f1_score'])
        elif sort_key == 'combined':
            winning_index = max(range(len(lod)), key=lambda index: lod[index]['combined_score'])
        else:
            raise Exception('Method does not exist yet.')

        return lod[winning_index]

    @staticmethod
    def write_to_csv(lod, filename = 'results.csv'):
        """
        Given a list of dictionaries, write the list to a csv
        """
        keys = lod[0].keys()
        with open(filename, 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(lod)
        logging.info('Results written to csv.')

    def grapple(self):
        """
        Wrapper function that loops through all possible levers, generates models for each, and runs tests
        """
        self.get_levers()
        running_total = 0

        for lever_set in self.lever_sets:
            retrieve_method = lever_set['retrieve_method']
            reader_method = lever_set['reader_method']

            for i, levers in enumerate(lever_set['levers']):
                # Generate answers and return dict with questions, answers, and scores
                answer_set = self.run_models(levers, retrieve_method, reader_method)
                # Generate score aggregates
                em_agg, f1_agg, combined = self.aggregate_scores(answer_set)
                print('\n\n')
                logging.info(f'Model {i + 1} of {self.total_permutations} complete.\n')
                logging.info(f"Using {retrieve_method} retriever and the {reader_method} framework and the following levers:\n")
                pprint.pprint(levers)
                print('\n\n')
                self.all_models_and_levers.append({
                    "batch_number": i,
                    "retrieve_method": retrieve_method,
                    "reader_method": reader_method,
                    "levers": levers,
                    "em_score": em_agg,
                    "f1_score": f1_agg,
                    "combined_score": combined
                })
                running_total += i
        winner = self.get_best_batch(self.all_models_and_levers)
        pprint.pprint(f"..and the winner is:\n\n{winner}")
        self.write_to_csv(self.all_models_and_levers)
