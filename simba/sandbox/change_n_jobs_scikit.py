import pickle

MODEL_PATH = r'/Users/simon/Downloads/Floor Skimming_3.sav'
NEW_MODEL_SAVE_PATH = r'/Users/simon/Downloads/Floor Skimming_3_new_n_jobs.sav'
NEW_N_JOBS = 16

clf = pickle.load(open(MODEL_PATH, "rb"))
clf.n_jobs = NEW_N_JOBS
pickle.dump(clf, open(NEW_MODEL_SAVE_PATH, "wb"))
