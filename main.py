import sys
print("[BOOT] main.py starting, Python imports beginning...", flush=True)
import os
print("[IMPORT] os ok", flush=True)
from dotenv import load_dotenv
load_dotenv()
print("[IMPORT] dotenv ok", flush=True)
import argparse
print("[IMPORT] argparse ok", flush=True)
from pipline_factory import LLMPipeline
print("[IMPORT] pipline_factory ok", flush=True)
from mylogger import Logger
print("[IMPORT] mylogger ok", flush=True)
from dataloader_factory import LLMDataLoader
print("[IMPORT] dataloader_factory ok", flush=True)


def get_args():
    parser = argparse.ArgumentParser()
    # LLM model name
    parser.add_argument('--model_type', type=str, default='llm', help='model type llm or ktm')
    parser.add_argument('--model_name', type=str, default='glm-4', help='model name')
    parser.add_argument('--data_path', type=str, default='./datasets', help='data path')
    # data_mode: sparse, moderate, rich
    parser.add_argument('--data_mode', type=str, default='sparse', help='data mode: onehot, sparse, moderate, rich')
    # dataset name
    parser.add_argument('--dataset_name', type=str, default='FrcSub', help='dataset name')
    parser.add_argument('--log_path', type=str, default='./logs', help='log path')
    # train_split
    parser.add_argument('--train_split', type=float, default=0.8, help='train split')
    # parser.add_argument('--is_shuffle', type=bool, default=True, help='shuffle data when splitting')
    parser.add_argument('--is_shuffle', action='store_true', help='shuffle data when splitting')
    # test number
    parser.add_argument('--test_num', type=int, default=20, help='test number')
    # specific student ids to run (comma-separated, e.g., "1,3,7")
    parser.add_argument('--student_ids', type=str, default=None,
                        help='Specific student IDs to run, comma-separated (e.g., "1,3,7"). '
                             'If specified, test_num is ignored and only these students are processed.')
    # random seed
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    # concurrent workers
    parser.add_argument('--workers', type=int, default=4, help='number of concurrent workers for student processing, set to 1 for sequential')

    # llm fewshot settings
    parser.add_argument('--fewshot_num', type=int, default=4, help='fewshot num, 0 means zero-shot')
    parser.add_argument('--fewshot_strategy', type=str, default='random', help='fewshot strategy, random/first/last/concept_based/knowledge_graph')
    parser.add_argument('--eval_strategy', type=str, default='simple', help='eval strategy, simple/analysis/self_correct')
    parser.add_argument(
        '--skip_post_explain',
        action='store_true',
        help='skip LLM call after prediction (generate_explaination); saves API cost, prediction path unchanged',
    )

    # knowledge graph settings
    parser.add_argument('--knowledge_graph_path', type=str, default=None,
                        help='Path to knowledge graph JSON file (for knowledge_graph strategy). '
                             'If not specified, will use default path based on dataset_name. '
                             'Example: ./datasets/moderate/MOOCRadar/knowledge_graph_correctness1.0_sequence0.0.json')
    parser.add_argument('--graph_config_name', type=str, default=None,
                        help='Short name for knowledge graph config (e.g., "correctness1.0_sequence0.0"). '
                             'Used in result filename. Auto-detected from knowledge_graph_path if not specified.')

    # version settings
    parser.add_argument('--version', type=str, default='v1',
                        help='Version of the pipeline: v1 (multi-turn) or v3 (producer-critic-judge). '
                             'v1: Multiple API calls for analysis, prediction, explanation. '
                             'v3: Producer-critic-judge pipeline with a v1-style producer.')

    args = parser.parse_args()
    return args


def main(args):
    print(f"[STEP 1] Logger init...", flush=True)
    my_logger = Logger(args)
    print(f"[STEP 2] Data path check: {args.data_mode}/{args.dataset_name}", flush=True)
    data_path = os.path.join(args.data_path, args.data_mode, args.dataset_name)
    if not os.path.exists(data_path):
        print(data_path)
        raise ValueError("data path not exist, check path, mode, or dataset name")
    if args.model_type == 'llm':
        print(f"[STEP 3] Loading user data (train/test split)...", flush=True)
        data_loader = LLMDataLoader(args=args, logger=my_logger)
        train_data, test_data = data_loader.load_user_data(data_path=data_path, train_split=args.train_split, is_shuffle=args.is_shuffle)
        print(f"[STEP 3 DONE] train={len(train_data)}, test={len(test_data)}", flush=True)
        print(f"[STEP 4] Loading exercise info ({args.data_mode})...", flush=True)
        if args.data_mode == 'onehot':
            extra_datas = data_loader.load_onehot_data(data_path=data_path)
        elif args.data_mode =='sparse':
            extra_datas = data_loader.load_sparse_data(data_path=data_path)
        elif args.data_mode =='moderate':
            extra_datas = data_loader.load_moderate_data(data_path=data_path)
        elif args.data_mode == 'rich':
            extra_datas = data_loader.load_rich_data()
        else:
            raise ValueError("data mode not in ['onehot','sparse','moderate', 'rich']")
        print(f"[STEP 4 DONE] exercise_info loaded", flush=True)
    elif args.model_type == 'ktm':
        # TODO: add ktm dataloader
        pass
    else:
        raise ValueError("model type not in ['llm', 'ktm']")

    print(f"[STEP 5] Initializing LLMPipeline (model={args.model_name}, version={args.version})...", flush=True)
    # initial pipline
    llm_pipline = LLMPipeline(model_name=args.model_name,
                              train_data=train_data,
                              test_data=test_data,
                              extra_datas=extra_datas,
                              logger=my_logger,
                              data_mode=args.data_mode,
                              fewshots_num=args.fewshot_num,
                              fewshots_strategy=args.fewshot_strategy,
                              eval_strategy=args.eval_strategy,
                              test_num=args.test_num,
                              random_seed=args.random_seed,
                              skip_post_explain=args.skip_post_explain,
                              dataset_name=args.dataset_name,
                              knowledge_graph_path=args.knowledge_graph_path,
                              max_workers=args.workers,
                              version=args.version,
                              )
    print(f"[STEP 6] Pipeline running...", flush=True)
    llm_pipline.run()
    print(f"[DONE] Pipeline finished.", flush=True)

if __name__ == "__main__":
    args = get_args()
    main(args)