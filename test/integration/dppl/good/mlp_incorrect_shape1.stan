
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
//    mlp.l1.weight ~  normal(zeros(mlp.l1.weight$shape), ones(mlp.l1.weight$shape));
    mlp.l1.weight ~  normal(zeros(), ones());
//    mlp.l1.bias ~ normal(zeros(mlp.l1.bias$shape), ones(mlp.l1.bias$shape));
    mlp.l1.bias ~ normal(zeros(), ones());
//    mlp.l2.weight ~ normal(zeros(mlp.l2.weight$shape), ones(mlp.l2.weight$shape));
    mlp.l2.weight ~ normal(zeros(), ones());
//    mlp.l2.bias ~  normal(zeros(mlp.l2.bias$shape), ones(mlp.l2.bias$shape));   //<- First argument has a different shape>
    mlp.l2.bias ~  normal(zeros(), ones());   //<- First argument has a different shape>

    logits = mlp(imgs);
    labels ~ categorical_logits(logits);
}

guide parameters {
    real l1wloc[_];
    real l1wscale[_];
    real l1bloc[_];
    real l1bscale[_];
    real l2wloc[_];
    real l2wscale[_];
    real l2bloc[_];
    real l2bscale[_];
}

guide {
    l1wloc = randn(); // l1wloc$shape);
    l1wscale = randn(); // l1wscale$shape);
    mlp.l1.weight ~  normal(l1wloc, softplus(l1wscale));
    l1bloc = randn(); // l1bloc$shape);
    l1bscale = randn(); // l1bscale$shape);
    mlp.l1.bias ~ normal(l1bloc, softplus(l1bscale));
    l2wloc = randn(); // l2wloc$shape);
    l2wscale = randn(); // l2wscale$shape);
    mlp.l2.weight ~ normal(l2wloc, softplus(l2wscale));
    l2bloc = randn(); // l2bloc$shape);
    l2bscale = randn(); // l2bscale$shape);
    mlp.l2.bias ~ normal(l1bloc, softplus(l2bscale));
}
