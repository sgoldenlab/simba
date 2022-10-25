import pandas as pd
from simba.misc_tools import detect_bouts

def find_frames_when_cue_light_on(data_df: pd.DataFrame,
                                  cue_light_names: list,
                                  fps: int,
                                  prior_window_frames_cnt: int,
                                  post_window_frames_cnt: int):
    light_on_dict = {}
    for cue_light in cue_light_names:
        light_on_dict[cue_light] = {}
        light_bouts = detect_bouts(data_df=data_df, target_lst=[cue_light], fps=fps)
        light_bouts['Start_pre_window'] = light_bouts['Start_frame'] - prior_window_frames_cnt
        light_bouts['Start_pre_window'] = light_bouts['Start_pre_window'].clip(lower=0)
        light_bouts['Start_post_window'] = light_bouts['End_frame']
        light_bouts['End_pre_window'] = light_bouts['Start_frame']
        light_bouts['End_post_window'] = light_bouts['End_frame'] + post_window_frames_cnt
        light_bouts['End_post_window'] = light_bouts['End_post_window'].clip(upper=len(data_df))
        light_on_dict[cue_light]['df'] = light_bouts
        light_on_dict[cue_light]['light_on_frames'] = list(data_df.index[data_df[cue_light] == 1])
        pre_window_frames, post_window_frames = [], []
        for i, r in light_bouts.iterrows():
            pre_window_frames.extend((list(range(r['Start_pre_window'], r['End_pre_window']))))
            post_window_frames.extend((list(range(r['Start_post_window'], r['End_post_window']))))
        light_on_dict[cue_light]['pre_window_frames'] = pre_window_frames
        light_on_dict[cue_light]['post_window_frames'] = post_window_frames

    return light_on_dict