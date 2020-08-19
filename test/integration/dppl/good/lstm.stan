
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
    real[] rnn.gru.weight_ih_l0;
    real[] rnn.gru.weight_hh_l0;
    real[] rnn.gru.bias_ih_l0;
    real[] rnn.gru.bias_hh_l0;
    real[] rnn.decoder.weight;
    real[] rnn.decoder.bias;
}

model {
    real logits[n_characters];
    rnn.encoder.weight ~  normal(zeros(rnn.encoder.weight$shape), ones(rnn.encoder.weight$shape));
    rnn.gru.weight_ih_l0 ~ normal(zeros(rnn.gru.weight_ih_l0$shape), ones(rnn.gru.weight_ih_l0$shape));
    rnn.gru.weight_hh_l0 ~ normal(zeros(rnn.gru.weight_hh_l0$shape), ones(rnn.gru.weight_hh_l0$shape));
    rnn.gru.bias_ih_l0 ~  normal(zeros(rnn.gru.bias_ih_l0$shape), ones(rnn.gru.bias_ih_l0$shape));
    rnn.gru.bias_hh_l0 ~  normal(zeros(rnn.gru.bias_hh_l0$shape), ones(rnn.gru.bias_hh_l0$shape));
    rnn.decoder.weight ~  normal(zeros(rnn.decoder.weight$shape), ones(rnn.decoder.weight$shape));
    rnn.decoder.bias ~  normal(zeros(rnn.decoder.bias$shape), ones(rnn.decoder.bias$shape));
    logits = rnn(input);
    category ~ categorical_logits(logits);
}

guide parameters {
    real ewl[rnn.encoder.weight$shape];
    real ews[rnn.encoder.weight$shape];
    real gw1l[rnn.gru.weight_ih_l0$shape];
    real gw1s[rnn.gru.weight_ih_l0$shape];
    real gw2l[rnn.gru.weight_hh_l0$shape];
    real gw2s[rnn.gru.weight_hh_l0$shape];
    real gb1l[rnn.gru.bias_ih_l0$shape];
    real gb1s[rnn.gru.bias_ih_l0$shape];
    real gb2l[rnn.gru.bias_hh_l0$shape];
    real gb2s[rnn.gru.bias_hh_l0$shape];
    real dwl[rnn.decoder.weight$shape];
    real dws[rnn.decoder.weight$shape];
    real dbl[rnn.decoder.bias$shape];
    real dbs[rnn.decoder.bias$shape];

}

guide {
    ewl = .001*randn(ewl$shape);
    ews = randn(ews$shape) -10.0;
    rnn.encoder.weight ~  normal(ewl, exp(ews));
    gw1l = .001*randn(gw1l$shape);
    gw1s = randn(gw1s$shape) -10.0;
    rnn.gru.weight_ih_l0 ~ normal(gw1l, exp(gw1s));
    gw2l = .001*randn(gw2l$shape);
    gw2s = randn(gw2s$shape) -10.0;
    rnn.gru.weight_hh_l0 ~ normal(gw2l, exp(gw2s));
    gb1l = .001*randn(gb1l$shape);
    gb1s = randn(gb1s$shape) -10.0;
    rnn.gru.bias_ih_l0 ~  normal(gb1l, exp(gb1s));
    gb2l = .001*randn(gb2l$shape);
    gb2s = randn(gb2s$shape) -10.0;
    rnn.gru.bias_hh_l0 ~  normal(gb2l, exp(gb2s));
    dwl = .001*randn(dwl$shape);
    dws = randn(dws$shape) -10.0;
    rnn.decoder.weight ~  normal(dwl, exp(dws));
    dbl = .001*randn(dbl$shape);
    dbs = randn(dbs$shape) -10.0;
    rnn.decoder.bias ~  normal(dbl, exp(dbs));
}
