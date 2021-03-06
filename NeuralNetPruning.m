%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Author : Saurabh Goyal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input to the Network
% - dataset_name = name of the dataset_name
% - num_labels = # of output classes
% NOTE : Each row of the dataset conatins a data point. The first column of dataset
%        contains the output labels starting from 1 and going upto # of classes.
%        Rest of the columns contains the features. The dataset should be comma separated.

% Hyper-parameters for grid search to find the best models within 16 KB. This will
% help in selecting the best model for a given memory limit by deciding the trade-off
% between having large number of hidden nodes with sparse connections or small number
% of nodes with dense connections

% - min_hidden_units = minimum # of nodes to start with
% - jumpsize_hidden_units = # of nodes to increase in each run
% - max_hidden_units = maximum # of nodes in the network
% - lambdaLR = Learning rate for the NN
% - pruning_percent = %age of weights to be pruned.
% - num_iter = # of iterations for gradient descent

% Output of the Network
% - theta1 = weight matrix from input layer to hidden layer
% - theta2 = weight matrix from hidden layer to output layer
% - model_size = model size of the sparse network in CSR Format. The non-zero value is
%                 stored as 32-bit float, column and row index as 16-bit integer.

% Files generated by the network
% - Output File = Contains the train and test accuracy along with model size for given
%                  set of hyper-parameters
% - Model Files = Contains sparse NN model in CSR format. The first 2 rows fo the Model
%                   file contains the # of non-zeros and size of row_idx for 2 set of
%                   theta parameters repectively

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dataset specfication
dataset_path = 'dataset/';
dataset_name = 'banana';
num_labels = 2;

output_file_path = ['OutputFile/'];
model_file_path = ['Models/' dataset_name '/'];
mkdir(output_file_path);
mkdir(model_file_path);


%% Hyper-parameters For Neural Network
min_hidden_units = 10;
max_hidden_units = 50;
jumpsize_hidden_units = 5;
lambdaLR = [0.001, 0.01, 0.1];
pruning_percent = [70 ,80, 90, 95];

%% Other parameters for the network
num_iter = 1000;


%% Loading train and test/validation set

dataTrain = dlmread([dataset_path dataset_name '.train'],',');
Xtrain = dataTrain(:,2:end);
Ytrain = dataTrain(:,1);

dataTest = dlmread([dataset_path dataset_name '.test'],',');
Xtest = dataTest(:,2:end);
Ytest = dataTest(:,1);

fout = fopen([output_file_path dataset_name '.txt'],'w');

for w1 = min_hidden_units : jumpsize_hidden_units : max_hidden_units

    for w2 = 1 : length(lambdaLR)

        for w3 = 1 : length(pruning_percent)


            %% =================== Training the NN  ===================== %%

            num_hidden_units = w1;
            perc_retain = 100 - pruning_percent(w3);
            lr = lambdaLR(w2);
            disp(['# of Hidden Nodes = ' num2str(num_hidden_units)]);
            disp(['Pruning Percent = ' num2str(pruning_percent(w3))]);
            disp(['Learning Rate = ' num2str(lr)]);
            [theta1, theta2] = train(Xtrain, Ytrain, num_labels, num_hidden_units, num_iter, lr, perc_retain);


            %% ================= Prediction on dataset ================== %%

            predtrain = predict(theta1, theta2, Xtrain);
            accuracyTrain = mean(double(predtrain == Ytrain)) * 100;
            fprintf('\nTrain Accuracy : %f\n', accuracyTrain);

            predtest = predict(theta1, theta2, Xtest);
            accuracyTest = mean(double(predtest == Ytest)) * 100;
            fprintf('\nTest Accuracy : %f\n', accuracyTest);


            %% ================= Model size calculation ================= %%

            %% Model size in kilobytes for model in CSR format.

            nnz_theta1 = theta1((theta1'~=0)');
            nnz_theta1_idx = find(theta1(1,:)~=0);
            row_idx_theta1 = length(nnz_theta1_idx);
            for num_rows = 2 : size(theta1,1)
                nnz_theta1_idx = [nnz_theta1_idx find(theta1(num_rows,:)~=0)];
                row_idx_theta1 = [row_idx_theta1 length(nnz_theta1_idx)];
            end
                                
            nnz_theta2 = theta2((theta2'~=0)');
            nnz_theta2_idx = find(theta2(1,:)~=0);
            row_idx_theta2 = length(nnz_theta2_idx);
            for num_rows = 2 : size(theta2,1)
                nnz_theta2_idx = [nnz_theta2_idx find(theta2(num_rows,:)~=0)];
                row_idx_theta2 = [row_idx_theta2 length(nnz_theta2_idx)];
            end
            
            size_layer1 = 4 * length(nnz_theta1) + 2 * (length(nnz_theta1_idx) + length(row_idx_theta1));
            size_layer2 = 4 * length(nnz_theta2) + 2 * (length(nnz_theta2_idx) + length(row_idx_theta2));
            modelsize = (size_layer1 + size_layer2)/1024;

            fprintf('\nModel Size : %f KB\n', modelsize);
            %% ============ Storing models and output file ============== %%

            fprintf(fout,'%s\n',[num2str(num_hidden_units) ',' num2str(lr) ',' num2str(pruning_percent(w3)) ',' num2str(modelsize) ',' num2str(accuracyTrain) ',' num2str(accuracyTest)]);
            
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], [length(nnz_theta1) size(theta1,1)],',');
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], [length(nnz_theta2) size(theta2,1)],'-append','delimiter',',');
                                
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], row_idx_theta1,'-append','delimiter',',');
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], nnz_theta1,'-append','delimiter',',');
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], nnz_theta1_idx,'-append','delimiter',',');
                                
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], row_idx_theta2,'-append','delimiter',',');
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], nnz_theta2,'-append','delimiter',',');
            dlmwrite([model_file_path dataset_name '_' num2str(num_hidden_units) '_' num2str(lr) '_' num2str(pruning_percent(w3)) '.txt'], nnz_theta2_idx,'-append','delimiter',',');
                
            disp('---------------------------')
        end
    end
end
fclose(fout);
