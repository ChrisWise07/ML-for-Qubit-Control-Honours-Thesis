for loss_name in "parameter_loss_mod" "combined_loss_mod"
do
    python3 ./main.py --model_name="simple_ANN" --num_epochs=2 --num_examples=500 --print_freq=5 --offline="True" --loss_name=$loss_name
done
