import os
import pandas as pd
import joblib
from sklearn import ensemble, preprocessing, metrics
from . import dispatcher



TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


if not TRAINING_DATA:
    raise ValueError("TRAINING_DATA environment variable is not set.")

if FOLD is None:
    raise ValueError("FOLD environment variable is not set.")

FOLD_MAPPING = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}



if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]
    
    # 
    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    # Create train and validation sets
    train_df = train_df.drop(["id","target","kfold"], axis=1)
    valid_df = valid_df.drop(["id","target","kfold"], axis=1)

    #
    valid_df = valid_df[train_df.columns]


    # Create encoded labels
    label_encoders = []
    for col in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[col].values.tolist()+valid_df[col].values.tolist())
        train_df.loc[:,col] = lbl.transform(train_df[col].values.tolist())
        valid_df.loc[:,col] = lbl.transform(valid_df[col].values.tolist())
        label_encoders.append((col,lbl))

    
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    
    print("Model Accuracy: ", metrics.roc_auc_score(yvalid, preds))

    # save label encoder and model
    joblib.dump(label_encoders, f"models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}.pkl")