
networks {
    RNN rnn;
}

data {
    int n_characters;
    int input[n_characters];
    int category[n_characters];
}

parameters {
    real[] rnn.encoder.weight;
}

model {
    int logits[n_characters];
    rnn.encoder.weight ~  normal(zeros(rnn.encoder.weight$shape), ones(rnn.encoder.weight$shape));
    logits = rnn(input);
    category ~ categorical_logits(logits);
}

guide parameters {
     real ewl[rnn.encoder.weight$shape];
     real ews[rnn.encoder.weight$shape];
}

guide {
     ewl = randn(ewl$shape);
     ews = randn(ews$shape) -10.0;
     rnn.encoder.weight ~  normal(ewl, exp(ews));
}
