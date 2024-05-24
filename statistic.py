import joblib
import os
import config

def load_exp_data():
    retrain_ans = joblib.load(os.path.join("exp_ans_data",config.dataset_name,config.model_name,"retrain_ans.data"))
    print("")

if __name__ == "__main__":
    load_exp_data()
