parameters {
    real y_std;
    real x_std;
}
transformed parameters{
    real y = 3.0 * y_std;
    real x = exp(y/2) * x_std;
}
model{
    y_std ~ normal(0, 1);
    x_std ~ normal(0, 1);
}