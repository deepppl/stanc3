networks {
  vector mlp(int[,,] imgs);
}

data {
    int batch_size;
    int nx;
    int nh;
    int ny;
    int <lower=0, upper=1> imgs[28,28,batch_size];
    int <lower=1, upper=10> labels[batch_size];
}

parameters {
    real mlp.l1.weight[nh, nx];
    real mlp.l1.bias[nh];
    real mlp.l2.weight[ny, nh];
    real mlp.l2.bias[ny];
}

model {
    vector[batch_size] logits;
    mlp.l1.weight ~  normal(0, 1);
    mlp.l1.bias ~ normal(0, 1);
    mlp.l2.weight ~ normal(0, 1);
    mlp.l2.bias ~  normal(0, 1);
    logits = mlp(imgs);
    labels ~ categorical_logit(logits);
}

guide parameters {
    real l1wloc[nh, nx];
    real l1wscale[nh, nx];
    real l1bloc[nh];
    real l1bscale[nh];
    real l2wloc[ny, nh];
    real l2wscale[ny, nh];
    real l2bloc[ny];
    real l2bscale[ny];
}

guide {
    mlp.l1.weight ~  normal(l1wloc, log1p_exp(l1wscale));
    mlp.l1.bias ~ normal(l1bloc, log1p_exp(l1bscale));
    mlp.l2.weight ~ normal(l2wloc, log1p_exp(l2wscale));
    mlp.l2.bias ~ normal(l2bloc, log1p_exp(l2bscale));
}
