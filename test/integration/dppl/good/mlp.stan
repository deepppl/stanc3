
networks {
    MLP mlp;
}

data {
    int batch_size;
    int <lower=0, upper=1> imgs[28,28,batch_size];
    int <lower=0, upper=10>  labels[batch_size];
    int mlp_l1_weight_shape;
    int mlp_l1_bias_shape;
    int mlp_l2_weight_shape;
    int mlp_l2_bias_shape;
}

parameters {
    real mlp.l1.weight[mlp_l1_weight_shape];
    real mlp.l1.bias[mlp_l1_weight_shape];
    real mlp.l2.weight[mlp_l1_weight_shape];
    real mlp.l2.bias[mlp_l1_weight_shape];
}

model {
    real logits[batch_size];
    mlp.l1.weight ~  normal(0, 1);
    mlp.l1.bias ~ normal(0, 1);
    mlp.l2.weight ~ normal(0, 1);
    mlp.l2.bias ~  normal(0, 1);

    logits = mlp(imgs);
    labels ~ categorical_logits(logits);
}

guide parameters {
    real l1wloc[mlp_l1_weight_shape];
    real l1wscale[mlp_l1_weight_shape];
    real l1bloc[mlp_l1_bias_shape];
    real l1bscale[mlp_l1_bias_shape];
    real l2wloc[mlp_l2_weight_shape];
    real l2wscale[mlp_l2_weight_shape];
    real l2bloc[mlp_l2_bias_shape];
    real l2bscale[mlp_l2_bias_shape];
}

guide {
    l1wloc = randn();
    l1wscale = randn();
    mlp.l1.weight ~  normal(l1wloc, softplus(l1wscale));
    l1bloc = randn();
    l1bscale = randn();
    mlp.l1.bias ~ normal(l1bloc, softplus(l1bscale));
    l2wloc = randn();
    l2wscale = randn();
    mlp.l2.weight ~ normal(l2wloc, softplus(l2wscale));
    l2bloc = randn();
    l2bscale = randn();
    mlp.l2.bias ~ normal(l2bloc, softplus(l2bscale));
}
