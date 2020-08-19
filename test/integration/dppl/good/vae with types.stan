
networks {
    Decoder decoder;
    Encoder encoder;
}

data {
    int nz; int batch_size;
    int<lower=0, upper=1> x[28, 28, batch_size];
}

parameters {
    real z[nz, batch_size];
}

model {
    real mu[28, 28, batch_size];
    // zeros : 'a -> [dimension='a, component='real]
    // Normal : [dimension='a, component='b = real/int] -> [dimension='a, component='b'] -> [dimension='a, component='b]
    // Normal : [dimension='a, component='b = real/int] -> [dimension='a, component='b'] -> int -> [dimension=int, component=[dimension='a, component='b]]
    // TODO: implement matrix for the second one.


    // Normal : 
    z ~ normal(zeros(nz), ones(nz));
    mu = decoder(z);
    x ~ bernoulli(mu);
}

guide {
    real encoded[2, nz, batch_size] = encoder(x);
    real mu_z[nz, batch_size] = encoded[1];
    real sigma_z[nz, batch_size] = encoded[2];
    z ~ normal(mu_z, sigma_z);
}
