export CUDA_VISIBLE_DEVICES='2'
python3.8 ./RelChoice/make_data.py -q $2 -c $1 -d ./cache -o rel && \
python3.8 ./RelChoice/predict.py --raw_test_file $2 --test_file ./cache/rel_0.json --target_dir ./RelChoice/saved/q4 --out_file ./cache/rel_out.json && \
python3.8 QuesAns/make_data.py -q ./cache/rel_out.json -c $1 -d ./cache -o qa && \
python3.8 QuesAns/predict.py --test_file ./cache/qa_0.json --target_dir ./QuesAns/saved/q4 --out_file $3
