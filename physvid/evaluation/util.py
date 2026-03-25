import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from physvid.util import get_sha256_key


def create_df(model_name, evaluator_name, eval_dir):
    dataset_name = evaluator_name
    caption_col = 'caption'
    threshold = 0.5 if evaluator_name == 'videophy' else 4
    split = 'test'
    version = dataset_name
    if version == 'videophy':
        dataset = f'videophysics/videophy_{split}_public'
    elif version == 'videophy2':
        dataset = f'videophysics/videophy2_{split}'

    df = load_dataset(dataset, split=split).to_pandas()

    # drop rows with empty or NaN captions and remove duplicates
    df[caption_col] = df[caption_col].str.strip().str.strip('"\'').str.strip()
    df = df[df[caption_col].notna() & (df[caption_col] != '')]
    df = df.drop_duplicates(subset=[caption_col], keep='first').reset_index(drop=True)

    for eval_index in range(5):
        df[f'sa{eval_index}'] = None
        df[f'pc{eval_index}'] = None

        suffix = '_short' if dataset_name == 'videophy2' else ''
        path = Path(f"{eval_dir}/eval{eval_index}/{evaluator_name}_{model_name}_results_{dataset_name}"
                    f"{suffix}.json")

        # continue if file does not exist
        if not path.exists():
            print(f"{path}: path does not exist")
            continue

        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        assert len(data) == len(df), "Data length does not match dataframe length!"
        for index, caption in df[caption_col].items():
            key = get_sha256_key(caption)
            entry = data.get(key, None)
            if entry is None:
                raise ValueError(f"Key {key} not found in data for caption: {caption}")

            # add entry['sa'] and entry['pc'] to df for this caption
            df.loc[index, f'sa{eval_index}'] = entry['sa']
            df.loc[index, f'pc{eval_index}'] = entry['pc']
            df.loc[index, f'jp{eval_index}'] = int(entry['sa'] >= threshold and entry['pc'] >= threshold)

    return df


def videophy_overall_results(df, th=.5):
    # task 1: average across sa0-sa4 and pc0-pc4 rowwise
    sa_means = df[[f'sa{i}' for i in range(5)]].ge(th).astype(int).mean(axis=0)
    pc_means = df[[f'pc{i}' for i in range(5)]].ge(th).astype(int).mean(axis=0)
    jp_means = df[[f'jp{i}' for i in range(5)]].mean(axis=0)

    # task 2: print average and stddev of means
    sa_mean = sa_means.mean()
    sa_std = sa_means.std()
    pc_mean = pc_means.mean()
    pc_std = pc_means.std()
    jp_mean = jp_means.mean()
    jp_std = jp_means.std()

    overall_df = pd.DataFrame([{
        'group': 'overall',
        'sa_mean': sa_mean, 'sa_std': sa_std,
        'pc_mean': pc_mean, 'pc_std': pc_std,
        'jp_mean': jp_mean, 'jp_std': jp_std,
    }])
    return overall_df


def videophy_grouped_results(cat1, cat2, evaluator_name, df):
    # task 3: groupby states_of_matter and apply videophy_overall_results function to each group
    th = 0.5 if evaluator_name == 'videophy' else 4

    # vectorized alternative to avoid groupby.apply
    cols_sa = [f"sa{i}" for i in range(5)]
    cols_pc = [f"pc{i}" for i in range(5)]
    cols_jp = [f"jp{i}" for i in range(5)]

    # group by cat1
    sa_g = df[cols_sa].ge(th).astype(int).groupby(df[cat1]).mean()
    pc_g = df[cols_pc].ge(th).astype(int).groupby(df[cat1]).mean()
    jp_g = df[cols_jp].groupby(df[cat1]).mean()

    grouped = pd.DataFrame({
        "group": sa_g.index,
        "sa_mean": sa_g.mean(axis=1).values,
        "sa_std": sa_g.std(axis=1).values,
        "pc_mean": pc_g.mean(axis=1).values,
        "pc_std": pc_g.std(axis=1).values,
        "jp_mean": jp_g.mean(axis=1).values,
        "jp_std": jp_g.std(axis=1).values,
    }).reset_index(drop=True)

    # group by cat2
    sa_g2 = df[cols_sa].ge(th).astype(int).groupby(df[cat2]).mean()
    pc_g2 = df[cols_pc].ge(th).astype(int).groupby(df[cat2]).mean()
    jp_g2 = df[cols_jp].groupby(df[cat2]).mean()

    complexity_df = pd.DataFrame({
        "group": sa_g2.index,
        "sa_mean": sa_g2.mean(axis=1).values,
        "sa_std": sa_g2.std(axis=1).values,
        "pc_mean": pc_g2.mean(axis=1).values,
        "pc_std": pc_g2.std(axis=1).values,
        "jp_mean": jp_g2.mean(axis=1).values,
        "jp_std": jp_g2.std(axis=1).values,
    }).reset_index(drop=True)

    complexity_df['group'] = complexity_df['group'].replace({1: 'hard', 0: 'easy'})
    return grouped, complexity_df


def overall_results(evaluator_name, df):
    if evaluator_name == 'videophy':
        return videophy_overall_results(df, th=.5)
    elif evaluator_name == 'videophy2':
        return videophy_overall_results(df, th=4)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")


def grouped_results(evaluator_name, df):
    if evaluator_name == 'videophy':
        return videophy_grouped_results('states_of_matter', 'complexity', evaluator_name, df)
    elif evaluator_name == 'videophy2':
        return videophy_grouped_results('category', 'is_hard', evaluator_name, df)
    else:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")


def results(eval_dir, evaluator_name, model_name):
    df = create_df(model_name, evaluator_name, eval_dir)

    overall_df = overall_results(evaluator_name, df)
    grouped, complexity_df = grouped_results(evaluator_name, df)
    long_results = pd.concat(
        [
            overall_df,
            grouped,
            complexity_df,
        ],
        ignore_index=True,
    )

    print(f"Results for {evaluator_name} - {model_name} - {eval_dir}")
    # only display mean columns
    display_cols = [col for col in long_results.columns
                    if 'mean' in col or col == 'group'
                    ]
    display_results = long_results[display_cols]
    print(display_results)
    return long_results
