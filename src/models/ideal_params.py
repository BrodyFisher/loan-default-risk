import json

best_params = {
    'n_estimators': 799,
    'learning_rate': 0.011667969103232043,
    'num_leaves': 32,
    'min_child_samples': 69,
    'feature_fraction': 0.8606930076507724,
    'bagging_fraction': 0.8366975307242142,
    'bagging_freq': 1,
    'reg_alpha': 5.820714180413196e-08,
    'reg_lambda': 2.8538023388998856e-06
}

with open('src/models/best_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print("Saved best params to src/models/best_params.json")