
networks {
    MLP mlp;
}

data {
    int batch_size;
    int <lower=0, upper=1> imgs[28,28,batch_size];
    int <lower=0, upper=10>  labels[batch_size];
}

parameters {
    real[] mlp.l1.weight;
    real[] mlp.l1.bias;
    real[] mlp.l2.weight;
    real[] mlp.l2.bias;
}

model {
    real logits[batch_size];
    mlp.l1.weight ~  normal(0, 1);
    mlp.l1.bias ~ normal(0, 1);
    mlp.l2.weight ~ normal(0, 1);
    mlp.l2.bias ~  normal(0, 1);
    logits = mlp(imgs);
    labels ~ categorical(logits);
}

guide parameters {
    real l1wloc;
    real l1wscale;
    real l1bloc;
    real l1bscale;
    real l2wloc;
    real l2wscale;
    real l2bloc;
    real l2bscale;
}

guide {
    l1wloc = randn(0,1);
    l1wscale = exp(randn());
    mlp.l1.weight ~  normal(l1wloc, l1wscale);
    l1bloc = randn(0,1);
    l1bscale = exp(randn());
    mlp.l1.bias ~ normal(l1bloc, l1bscale);
    l2wloc = randn(0,1);
    l2wscale = exp(randn());
    mlp.l2.weight ~ normal(l2wloc, l2wscale);
    // l2bloc = randn();
    // l2bscale = exp(randn());
    // mlp.l2.bias ~ normal(l2bloc, l2bscale);
}
