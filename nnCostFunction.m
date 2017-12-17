function [J grad] = nnCostFunction(nn_params, input_layer_size, num_hidden_units, num_labels, XTrain, YTrain, lr, Theta1_counter, Theta2_counter, retrain_flag)

    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:num_hidden_units * (input_layer_size + 1)), num_hidden_units, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (num_hidden_units * (input_layer_size + 1))):end), num_labels, (num_hidden_units + 1));

    % Number of training examples
    m = size(XTrain, 1);

    %% ================================== Forward Pass for NN ================================== %%

    % Add a bias term to the input layer
    XTrain = [ones(m,1) XTrain];

    % Compute Activation value for the hidden layer
    Z2 = XTrain * Theta1';
    A2 = 1.0 ./ (1.0 + exp(-Z2));
    A2 = [ones(m,1) A2];

    % Compute Activation value for the output layer
    Z3 = A2*Theta2';
    A3 = 1.0 ./ (1.0 + exp(-Z3));

    % Convert numeric labels y to bit value Y
    Ytrain = zeros(m,num_labels);
    for i = 1 : m
        colindex = YTrain(i,1);
        Ytrain(i,colindex) = 1;
    end

    % Compute the Logistic Loss for the given thetas
    Err = -(1/m) * (Ytrain.*log(A3) + (1-Ytrain).*log(1-A3));

    % Compute the Loss function along with regularisation
    J = sum(sum(Err)) + (lr/(2*m)) * (sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));


    %% ================================== Backward Pass for NN ================================== %%

    % Calculate Residual error of the ouput Layer
    del3 = A3 - Ytrain;

    % Calculate Residual error of the hidden Layer
    del2 = del3 * Theta2;
    del2 = del2(:,2:end).*(1.0 ./ (1.0 + exp(-Z2)).*(1-1.0 ./ (1.0 + exp(-Z2))));

    % Calculate Gradient of the Thetas of input and hidden layer
    Theta1_grad = (del2' * XTrain)/m + (lr/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)] ;
    Theta2_grad = (del3' * A2)/m + (lr/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)] ;

    if(retrain_flag == true)
        Theta1_grad = Theta1_grad.*Theta1_counter;
        Theta2_grad = Theta2_grad.*Theta2_counter;
    end
                   
    %% ================================== Gradient Update for NN ================================== %%
                   
    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
               
end
