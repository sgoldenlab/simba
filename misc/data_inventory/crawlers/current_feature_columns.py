"""Run this repo's 16 and 14 body-part feature extractors on random pose data in a throwaway project and save the
output column names (pose + features, in order) -> data/current_feature_columns.json.
The page compares every training dataset against these to flag renamed / missing feature columns."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from simba.feature_extractors.feature_extractor_14bp import ExtractFeaturesFrom14bps  # noqa: E402
from simba.feature_extractors.feature_extractor_16bp import ExtractFeaturesFrom16bps  # noqa: E402
from simba.utils.config_creator import ProjectConfigCreator  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'current_feature_columns.json')
CONFIG_NAMES = os.path.join(REPO, 'simba', 'pose_configurations', 'configuration_names', 'pose_config_names.csv')
names = [l.strip() for l in open(CONFIG_NAMES, encoding='utf-8') if l.strip()]

out = {}
tmp = tempfile.mkdtemp()   # not TemporaryDirectory: SimBA keeps the project log open, so cleanup is best-effort
for bp, extractor in (('16', ExtractFeaturesFrom16bps), ('14', ExtractFeaturesFrom14bps)):
    config_path = ProjectConfigCreator(project_path=os.path.join(tmp, bp), project_name='p', target_list=['Behavior'], pose_estimation_bp_cnt=bp,
                                       body_part_config_idx=names.index(f'2 animals; {bp} body-parts'), animal_cnt=2, file_type='csv').config_path
    project = os.path.dirname(config_path)
    bps = pd.read_csv(os.path.join(project, 'logs', 'measures', 'pose_configs', 'bp_names', 'project_bp_names.csv'), header=None)[0].tolist()
    cols = [f'{b}_{c}' for b in bps for c in ('x', 'y', 'p')]
    pose = pd.DataFrame(np.random.default_rng(0).uniform(50, 500, (300, len(cols))), columns=cols)
    pose[[c for c in cols if c.endswith('_p')]] = 0.9
    os.makedirs(os.path.join(project, 'csv', 'outlier_corrected_movement_location'), exist_ok=True)
    pose.to_csv(os.path.join(project, 'csv', 'outlier_corrected_movement_location', 'v1.csv'))
    pd.DataFrame([{'Video': 'v1', 'fps': 30, 'Resolution_width': 640, 'Resolution_height': 480, 'Distance_in_mm': 100, 'pixels/mm': 2.0}]) \
        .to_csv(os.path.join(project, 'logs', 'video_info.csv'), index=False)
    extractor(config_path=config_path).run()
    out[bp] = list(pd.read_csv(os.path.join(project, 'csv', 'features_extracted', 'v1.csv'), index_col=0, nrows=1).columns)
    print(bp, 'body parts:', len(out[bp]), 'columns')
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
shutil.rmtree(tmp, ignore_errors=True)
