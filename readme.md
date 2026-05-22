TO Make the run again use force flag.

python generate_result_force.py --force
python compute_train_auc_plot.py --force

STEPS are:
parse_data.py
fact_checker.py
generate_result.py
compute_train_auc.py

For small dataset:
python generate_result.py --test data/small_test.nt --out output/small_result.ttl