data {
}
parameters {
  real theta;
}
model {
    target += -0.5*(theta-1000.0)*(theta-1000.0);
}
