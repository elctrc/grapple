# Grapple
Grapple is a testing framework for the Haystack QA Framework. The intention of the framework is to allow a user to build a configuration file with all possible parameters and then run through a predetermined list of questions using every possible permutation of those parameters, returning F1 and exact match scores for each run.

## Installation
Grapple comes pre-loaded with Haystack as a Submodule. As such, installation is a tad different than usual.

### Clone Method
This will install all requirements for Grapple, but not Grapple itself - which you can access as you normally would via the file structure (see below)

#### Using Virtualenv
```bash
# This will install Grapple along with the Haystack submodule
> git clone --recurse-submodules INSERT PATH TO GITHUB PROJECT
> cd grapple
# Set up virtual environment
> virtualenv venv
> source venv/bin/activate
# This will install YAML and any other future dependencies
> pip install -r requirements
> cd haystack
# This will install the Haystack submodule
> pip install --editable .
```

#### Using Pipenv
__This method is currently NOT working__
```bash
> git clone --recurse-submodules INSERT PATH TO GITHUB PROJECT
> cd grapple
# Set up virtual environment
> pipenv shell
# This will install YAML and any other future dependencies
> pipenv install
> cd haystack
# This will install the Haystack submodule
> pip install --editable .
```

### pip Method
__This method is currently NOT working__
This is not currently working. In the future this _should_ work:

```bash
> pip install -e git+INSERT PATH TO GITHUB PROJECT#egg=Grapple
```

## Using the Package
The first thing you'll need to do is make a copy of `config_sample.yaml`. Rename to `config.yaml` and update your parameter/lever settings to whatever you would like to test (this filename can be anything but if you want to run with the default, use `config.yaml`.)

Secondly, you'll need to generate your test question set. Make a copy of `test_set_sample.yaml` and update with your questions and ground truth responses. Again, to use the defaults, save this new file as `test_set.yaml`.

From there, the easiest way to use Grapple is to simply run `run.py`. If you use the default filenames you will not need to use any other optinal command-line arguments.

__Note:__ You may want to install coloredlogs, which will make viewing the logs much clearer:

```bash
> pip install coloredlogs
```

Once installed, you only need to un-comment lines 2 and 5 in `run.py` and it will work!

Grapple will now run through each possible iteration of parameters, returning a csv named `results.csv` with the results.

If you do not want to use `run.py` and would like to tweak things on your own, all you need to do is run the following, replacing the config/questions with your custom yaml files:
```python
hl = HayLoader(config=config.yaml, test_set=questions.yaml)
hl.grapple()
```

## Notes
### Scoring
The results csv file contains seven columns:
* batch_number: This number represents the index of the parameter set used. This index resets with each reader_method
* retrieve_method: The retrieve method used (currently either `tfidf` or `embeddings`)
* reader_method: The reader method used (currently either `farm` or `transformers`)
* levers: A dictionary containing the exact levers used for this permutation
* em_score: The exact match average score for this run of questions (exact match score will be either 1 or 0 and the average is extrapolated from there)
* f1_score: The average f1 score for this run of questions
* combined_score: The em and f1 scored added together

### Garbage Collection
When you run Grapple, it will generate two local folders: `data` (or whatever you have specified as the `destination_dir` in config.yaml) which will contain your text sources and `mlruns` which contains Haystack metadata. With each run of Grapple, if you do not dump the folder structure it will not attempt to overwrite with new data, assuming your source is the same.

### models.yaml
This is a very loose collection of models grabbed from https://huggingface.co/models as a starting point for model selection. The file is not piped anywhere in the source code. It is solely here (for now anyway) as a reference only.

### What's Next
We still need to set up all the methods for using Elasticsearch as the store_type. Once that is done, using "embeddings" as a retrieve_method _should_ just work as the method is already set up.
