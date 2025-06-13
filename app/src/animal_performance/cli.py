
import argparse
from core import batch_map_via_queries, get_animal_performance
from data import add_data, choose_path
import yaml

def main():
    parser = argparse.ArgumentParser(description="NSK2 CLI")
    subparsers = parser.add_subparsers(dest='command')

    # Subcommand: animal-perf
    ap_parser = subparsers.add_parser('get_animal_performance', help='Run Animal Performace Analysis')
    ap_parser.add_argument('--mongo-uri', default= "mongodb://localhost:27017", help='MongoDB URI')
    # ap_parser.add_argument('--input-dir', required=True, help='Path to input netcdf files')
    # ap_parser.add_argument('--tmpdir', required=True, help='Directory to save mongodao and filesystemdao') 
    # ap_parser.add_argument('--propertymodels-dir', required=True, help='Path to input property models')
    # ap_parser.add_argument('--metamodels-dir', required=True, help='Path to input meta models')
    # ap_parser.add_argument('--datamodels-dir', required=True, help='Path to input data models')
    # ap_parser.add_argument('--settings', required=True, help='Path to settings.yaml file')
    # ap_parser.add_argument('--query', required=True, help='Path to query.yaml file')

    # input_dir = choose_path("input")
    # output_dir = choose_path("output")
    # tmpdir = choose_path("internal")
    # propertymodels_dir = choose_path("property models")
    # metamodels_dir = choose_path("meta models")
    # datamodels_dir = choose_path("data models")
    # settings = choose_path("settings.yml file")
    # query = choose_path("query.yml file")
    input_dir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/data/input"
    output_dir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/data/output"
    tmpdir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/data/internal"
    propertymodels_dir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/tests/data/valid_data/models/property_models.json"
    metamodels_dir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/tests/data/valid_data/models/metamodels"
    datamodels_dir = "/Users/Anurag/Downloads/NeuroLab/SignalStore/tests/data/valid_data/models/data_models"
    settings = "/Users/Anurag/Downloads/NeuroLab/SignalStore/app/src/animal_performance/queries_and_settings/settings.yml"
    query = "/Users/Anurag/Downloads/NeuroLab/SignalStore/app/src/animal_performance/queries_and_settings/query.yml"


    # Subcommand: nuerofunc
    neurofunc_parser = subparsers.add_parser('neurofunc', help='Run Animal Performace Analysis')
    neurofunc_parser.add_argument('--input-dir', required=True, help='Path to input netcdf files')
    neurofunc_parser.add_argument('--tmpdir', required=True, help='Directory to save mongodao and filesystemdao') 
    neurofunc_parser.add_argument('--propertymodels-dir', required=True, help='Path to input property models')
    neurofunc_parser.add_argument('--metamodels-dir', required=True, help='Path to input meta models')
    neurofunc_parser.add_argument('--datamodels-dir', required=True, help='Path to input data models')

    args = parser.parse_args()

    analysis_map = {
    "get_animal_performance": get_animal_performance
    }
    analysis_fn = analysis_map[args.command if args.command else "get_animal_performance"]

    unit_of_work = add_data(
        # args.mongo_uri,
        "mongodb://localhost:27017",
        input_dir,
        tmpdir,
        propertymodels_dir,
        metamodels_dir,
        datamodels_dir
    )

    
    with open(settings, "r") as f:
        settings_dict = yaml.safe_load(f)

    with open(query, "r") as f:
        query = yaml.safe_load(f)

    with unit_of_work as uow:
        sorted_by = [("session_start", -1)]
        session_queries = uow.data.find(query)
        batch_map_via_queries(
            uow,
            session_queries,
            settings_dict, 
            output_dir,
            analysis_fn,
            analysis_name=args.command if args.command else "get_animal_performance",)

if __name__ == "__main__":
    main()