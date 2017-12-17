function p = predict(Theta1, Theta2, X)

    % Number of Training Examples
    m = size(X, 1);

    h1 = 1.0 ./ (1.0 + exp(-[ones(m, 1) X] * Theta1'));
    h2 = 1.0 ./ (1.0 + exp(-[ones(m, 1) h1] * Theta2'));
    [dummy, p] = max(h2, [], 2);
                 
end
