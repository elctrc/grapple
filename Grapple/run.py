import argparse
# import coloredlogs
from grapple import HayLoader

# coloredlogs.install(level='DEBUG')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_sample.yaml', help='Path to config file')
    parser.add_argument('--questions', default='test_set_sample.yaml', help='Path to questions file')
    args = parser.parse_args()

    hl = HayLoader(config=args.config, test_set=args.questions)
    hl.grapple()


if __name__ == "__main__":
    main()