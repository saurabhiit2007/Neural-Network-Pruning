function [Theta1, Theta2] = train(Xtrain, Ytrain, num_labels, num_hidden_units, num_iter, lr,  perc_retain)


    %% ========================= Initializing Pameters ========================== %%

    retrain_flag = false;

    % Randomly initialize Theta(s) from -Epsilon to +Epsilon
    epsilon_init = 0.1;
    input_layer_size = size(Xtrain,2);
    initial_Theta1 = rand(num_hidden_units, 1 + input_layer_size) * 2 * epsilon_init - epsilon_init;
    initial_Theta2 = rand(num_labels, 1 + num_hidden_units) * 2 * epsilon_init - epsilon_init;

    Theta1_counter = ones(size(initial_Theta1,1),size(initial_Theta1,2));
    Theta2_counter = ones(size(initial_Theta2,1),size(initial_Theta2,2));

    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    disp('Weights initialization Done !!')

    %% ========================= Initial Training of NN ========================= %%

    options = optimset('MaxIter', num_iter);
    costFunction = @(p) nnCostFunction(p, input_layer_size, num_hidden_units, num_labels, Xtrain, Ytrain, lr, Theta1_counter, Theta2_counter, retrain_flag);

    % Use matlab function fmincg to optimize the Theta(s)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    disp('Initial Training Done !!')
    %% ============================= Pruning NN ================================= %%

    nn_params_sorted = sort(abs(nn_params),'descend');
    cutoff = nn_params_sorted(floor(perc_retain * size(nn_params,1)/100),1);

    nn_params_counter = [Theta1_counter(:) ; Theta2_counter(:)];
    for w=1:size(nn_params,1)
        if(abs(nn_params(w,1)) < cutoff)
            nn_params(w,1) = 0.0;
            nn_params_counter(w,1) = 0;
        end
    end
    Theta1_counter = reshape(nn_params(1:num_hidden_units * (input_layer_size + 1)),num_hidden_units, (input_layer_size + 1));
    Theta2_counter = reshape(nn_params_counter((1 + (num_hidden_units * (input_layer_size + 1))):end), num_labels, (num_hidden_units + 1));

    disp('Pruning Done !!')
    %% ============================= Retraining of NN after pruning ============== %%

    retrain_flag = true;

    costFunction = @(p) nnCostFunction(p, input_layer_size, num_hidden_units, num_labels, Xtrain, Ytrain, lr, Theta1_counter, Theta2_counter, retrain_flag);

    % Use inbuilt Function fmincg to optimize the Theta(s)
    [nn_params, cost] = fmincg(costFunction, nn_params, options);
    Theta1 = reshape(nn_params(1:num_hidden_units * (input_layer_size + 1)),num_hidden_units, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (num_hidden_units * (input_layer_size + 1))):end), num_labels, (num_hidden_units + 1));

    disp('Retraining Done !!')

end
