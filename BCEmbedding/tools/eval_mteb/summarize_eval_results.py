import argparse
import json
import os
import os.path as osp
from collections import defaultdict

from BCEmbedding.evaluation import c_mteb
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('evaluation.summarize_eval_results')


def read_results(task_types, except_tasks, args):
    tasks_results = {}
    model_dirs = {}
    for lang in args.lang:
        tasks_results[lang] = {}
        for t_type in task_types:
            if 'zh' == lang:
                mteb_tasks = c_mteb.MTEB(task_types=[t_type], task_langs=[lang, 'zh-CN']).tasks
            else:
                mteb_tasks = c_mteb.MTEB(task_types=[t_type], task_langs=[lang]).tasks

            if len(mteb_tasks) == 0: continue
                
            tasks_results[lang][t_type] = {}
            for t in mteb_tasks:
                task_name = t.description["name"]
                if task_name in except_tasks: continue

                models_names = os.listdir(args.results_dir)
                models_names.sort()
                if len(models_names) == 0: continue

                metric = t.description["main_score"]
                for model_name in models_names:
                    model_dir = os.path.join(args.results_dir, model_name)
                    if not os.path.isdir(model_dir): continue
                    
                    model_dirs[model_name] = model_dir
                    if os.path.exists(os.path.join(model_dir, task_name + '.json')):
                        if task_name not in tasks_results[lang][t_type]:
                            tasks_results[lang][t_type][task_name] = defaultdict(None)
                        with open(os.path.join(model_dir, task_name + '.json')) as f:
                            data = json.load(f)
                        for s in ['test', 'dev', 'validation', 'dev2']:
                            if s in data:
                                split = s
                                break

                        if 'en' == lang:
                            if 'en-en' in data[split]:
                                temp_data = data[split]['en-en']
                            elif 'en' in data[split]:
                                temp_data = data[split]['en']
                            else:
                                temp_data = data[split]
                        elif 'zh' == lang:
                            if 'zh-zh' in data[split]:
                                temp_data = data[split]['zh-zh']
                            elif 'zh' in data[split]:
                                temp_data = data[split]['zh']
                            elif 'zh-CN' in data[split]:
                                temp_data = data[split]['zh-CN']
                            else:
                                temp_data = data[split]
                        elif 'en-zh' == lang:
                            if 'en-zh' in data[split]:
                                temp_data = data[split]['en-zh']
                            elif 'en-zh-CN' in data[split]:
                                temp_data = data[split]['en-zh-CN']
                            else:
                                temp_data = data[split]
                        elif 'zh-en' == lang:
                            if 'zh-en' in data[split]:
                                temp_data = data[split]['zh-en']
                            elif 'zh-CN-en' in data[split]:
                                temp_data = data[split]['zh-CN-en']
                            else:
                                temp_data = data[split]

                        if metric == 'ap':
                            tasks_results[lang][t_type][task_name][model_name] = temp_data['cos_sim']['ap'] * 100
                        elif metric == 'cosine_spearman':
                            tasks_results[lang][t_type][task_name][model_name] = temp_data['cos_sim']['spearman'] * 100
                        else:
                            tasks_results[lang][t_type][task_name][model_name] = temp_data[metric] * 100

    return tasks_results, model_dirs


def output_markdown(tasks_results_wt_langs, model_names, model_type, save_file):
    with open(save_file, 'w') as f:
        f.write(f"# {model_type} Evaluation Results  \n")
        task_type_res_merge_lang = {}
        has_CQADupstack_overall = False
        for lang, tasks_results in tasks_results_wt_langs.items():
            f.write(f'## Language: `{lang}`  \n')

            task_type_res = {}
            for t_type, type_results in tasks_results.items():
                has_CQADupstack = False
                task_cnt = 0

                tasks_names = list(type_results.keys())
                if len(tasks_names) == 0:
                    continue

                task_type_res[t_type] = defaultdict()
                if t_type not in task_type_res_merge_lang:
                    task_type_res_merge_lang[t_type] = defaultdict(list)
                    
                f.write(f'\n### Task Type: {t_type}  \n')
                first_line = "| Model |"
                second_line = "|:-------------------------------|"
                for task_name in tasks_names:
                    if "CQADupstack" in task_name:
                        has_CQADupstack = True
                        has_CQADupstack_overall = True
                        continue
                    first_line += f" {task_name} |"
                    second_line += ":--------:|"
                    task_cnt += 1
                if has_CQADupstack:
                    first_line += f" CQADupstack |"
                    second_line += ":--------:|"
                    task_cnt += 1
                f.write(first_line + ' Avg |  \n')
                f.write(second_line + ':--------:|  \n')

                for model in model_names:
                    write_line = f"| {model} |"
                    all_res = []
                    cqa_res = []
                    for task_name in tasks_names:
                        results = type_results[task_name]
                        if "CQADupstack" in task_name:
                            if model in results:
                                cqa_res.append(results[model])
                            continue

                        if model in results:
                            write_line += " {:.2f} |".format(results[model])
                            all_res.append(results[model])
                        else:
                            write_line += f"  |"

                    if len(cqa_res) > 0:
                        write_line += " {:.2f} |".format(sum(cqa_res) / len(cqa_res))
                        all_res.append(sum(cqa_res) / len(cqa_res))

                    if len(all_res) == task_cnt:
                        write_line += " {:.2f} |".format(sum(all_res) / max(len(all_res), 1))
                        task_type_res[t_type][model] = all_res

                        task_type_res_merge_lang[t_type][model].extend(all_res)

                    else:
                        write_line += f"  |"
                    f.write(write_line + '  \n')

            f.write(f'\n### *Summary on `{lang}`*  \n')
            first_line = "| Model |"
            second_line = "|:-------------------------------|"
            task_type_res_keys = list(task_type_res.keys())
            for t_type in task_type_res_keys:
                first_line += f" {t_type} |"
                second_line += ":--------:|"
            f.write(first_line + ' Avg |  \n')
            f.write(second_line + ':--------:|  \n')

            for model in model_names:
                write_line = f"| {model} |"
                all_res = []
                for type_name in task_type_res_keys:
                    results = task_type_res[type_name]
                    if model in results:
                        write_line += " {:.2f} |".format(sum(results[model]) / max(len(results[model]), 1))
                        all_res.extend(results[model])
                    else:
                        write_line += f"  |"

                if len(all_res) > 0:
                    write_line += " {:.2f} |".format(sum(all_res) / len(all_res))

                f.write(write_line + '  \n')
        

        f.write(f'## Summary on all langs: `{list(tasks_results_wt_langs.keys())}`  \n')
        first_line = "| Model |"
        second_line = "|:-------------------------------|"
        task_type_res_merge_lang_keys = list(task_type_res_merge_lang.keys())
        task_nums = 0
        for t_type in task_type_res_merge_lang_keys:
            task_num = max([len(tmp_metrics) for tmp_model_name, tmp_metrics in task_type_res_merge_lang[t_type].items()])
            if t_type == 'Retrieval' and has_CQADupstack_overall: # 'CQADupstack' has been merged 12 to 1. We should add 11.
                task_num +=  11
            task_nums += task_num
            first_line += f" {t_type} ({task_num}) |"
            second_line += ":--------:|"
        f.write(first_line + f' Avg ({task_nums}) |  \n')
        f.write(second_line + ':--------:|  \n')

        for model in model_names:
            write_line = f"| {model} |"
            all_res = []
            for type_name in task_type_res_merge_lang_keys:
                results = task_type_res_merge_lang[type_name]
                if model in results:
                    write_line += " {:.2f} |".format(sum(results[model]) / max(len(results[model]), 1))
                    all_res.extend(results[model])
                else:
                    write_line += f"  |"

            if len(all_res) > 0:
                write_line += " {:.2f} |".format(sum(all_res) / len(all_res))

            f.write(write_line + '  \n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default=osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "results/embedding_results"), type=str, help="eval results path")
    parser.add_argument('--model_type', default="embedding", choices=['embedding', 'reranker'], type=str, help="model type, including `embedding` and `reranker` models")
    parser.add_argument('--lang', default="en-zh", choices=['en', 'zh', 'en-zh', 'zh-en'], type=str, help="choice which language eval results you want to collecte.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    save_name = osp.basename(args.results_dir.strip('/'))
    if 'embedding' in save_name.lower():
        args.model_type = 'embedding'
        save_name = 'embedding_eval_summary'
    elif 'reranker' in save_name.lower():
        args.model_type = 'reranker'
        save_name = 'reranker_eval_summary'
    
    if args.lang == 'zh':
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['zh', 'zh-CN']
    elif args.lang == 'en':
        task_types = ["Retrieval", "Clustering", "PairClassification", "Reranking", "STS", # "Summarization",
                      "Classification"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['en']
    elif args.lang in ['en-zh', 'zh-en']:
        task_types = ["Retrieval", "STS", "PairClassification", "Classification", "Reranking", "Clustering"]
        except_tasks = ['MSMARCOv2', 'BUCC']  # dataset is too large
        args.lang = ['en', 'zh', 'en-zh', 'zh-en']
    else:
        raise NotImplementedError(f"args.lang must be zh or en, but {args.lang}")
    
    args.model_type = args.model_type.capitalize()
    logger.info(f'eval results path: {args.results_dir}')
    logger.info(f'model type: {args.model_type}')
    logger.info(f'collect languages: {args.lang}')

    task_results, model_dirs = read_results(task_types, except_tasks, args=args)

    output_markdown(
        task_results, 
        model_dirs.keys(),
        args.model_type,
        save_file=os.path.join(
            args.results_dir, 
            save_name + ".md"
            )
        )
