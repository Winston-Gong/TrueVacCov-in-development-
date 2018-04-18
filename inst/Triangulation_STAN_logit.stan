data {
  int<lower=0> N; // number of subjects 
  int<lower=1, upper=3> Ab[N]; // biomarker
  int<lower=0, upper=1> F1[N]; // factor 1 
  int<lower=0, upper=1> F2[N]; // factor 2  
  int<lower=0, upper=1> F3[N]; // factor 3
  int<lower=0, upper=1> F4[N]; // factor 4  
  int<lower=0, upper=1> F5[N]; // factor 5  
}
parameters {
  real<lower=0, upper=1> thetavar0ab2; //prop Ab+ given neg vac
  real<lower=0, upper=1> thetavar0ab1ofno2; //prop Ab- given not Ab+ & neg vac
  real<lower=0, upper=1> thetavar1ab2; //prop Ab+ given true pos vac
  real<lower=0, upper=1> thetavar1ab1ofno2; //prop Ab- given not Ab+ & pos vac
  real<lower=0, upper=0.10> tau1; // restrain to small
  real<lower=0, upper=1> tau2;
  real<lower=0, upper=1> tau3;
  real<lower=0, upper=1> tau4;
  real<lower=0.9, upper=1> tau5; // restrain to large
}
transformed parameters {
  real<lower=0, upper=1> tau[N];
  vector<lower=0, upper=1>[3] thetavar0; //vector theta given true neg vac
  vector<lower=0, upper=1>[3] thetavar1; //vector theta given true pos vac
  for (n in 1:N) {
    tau[n]=tau1*F1[n]+tau2*F2[n]+tau3*F3[n]+tau4*F4[n]+tau5*F5[n];
  }
  thetavar0[1]=(1-thetavar0ab2)*thetavar0ab1ofno2; //prop Ab- given neg vac
  thetavar0[2]=thetavar0ab2; // prop Ab+ given neg vac
  thetavar0[3]=1-thetavar0[1]-thetavar0[2]; // prop Abe given neg vac
  thetavar1[1]=(1-thetavar1ab2)*thetavar1ab1ofno2; //prop Ab- given neg vac
  thetavar1[2]=thetavar1ab2; // prop Ab+ given neg vac
  thetavar1[3]=1-thetavar1[1]-thetavar1[2]; // prop Abe given neg vac
}
model {
  for (n in 1:N) {
    real lp[2]; //two log probability, one for when var=0 and one for =1 
    lp[1]=bernoulli_lpmf(0|tau[n])+categorical_lpmf(Ab[n]|thetavar0);
    lp[2]=bernoulli_lpmf(1|tau[n])+categorical_lpmf(Ab[n]|thetavar1);
    target += log_sum_exp(lp);
  }
}
generated quantities {
  real<lower=0, upper=1> Vac[N];
  real<lower=0, upper=1> Vacbioall;
  for (n in 1:N) {
    Vac[n]=bernoulli_rng(tau[n]);
  }
  Vacbioall=mean(Vac);
}
