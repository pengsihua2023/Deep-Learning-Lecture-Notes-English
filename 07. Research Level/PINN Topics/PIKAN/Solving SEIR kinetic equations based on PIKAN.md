# Solving SEIR Dynamical Equations Based on PIKAN

## 1. SEIR Dynamical Equations

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta S I, \\
\frac{dE}{dt} &= \beta S I - \sigma E, \\
\frac{dI}{dt} &= \sigma E - \gamma I, \\
\frac{dR}{dt} &= \gamma I.
\end{aligned}
$$

## 2. Neural Network Approximate Solution

Assume the PIKAN / PINN network outputs are:

$$
\hat{S}(t), \hat{E}(t), \hat{I}(t), \hat{R}(t).
$$

And their derivatives are obtained via automatic differentiation:

$$
\frac{d\hat{S}}{dt},\quad \frac{d\hat{E}}{dt},\quad \frac{d\hat{I}}{dt},\quad \frac{d\hat{R}}{dt}.
$$

## 3. Definition of Equation Residuals

Substitute the predicted values into the SEIR equations to obtain residuals:

$$
\begin{aligned}
r_S(t) &= \frac{d\hat{S}}{dt} + \beta \hat{S}(t)\hat{I}(t), \\
r_E(t) &= \frac{d\hat{E}}{dt} - \big(\beta \hat{S}(t)\hat{I}(t) - \sigma \hat{E}(t)\big), \\ 
r_I(t) &= \frac{d\hat{I}}{dt} - \big(\sigma \hat{E}(t) - \gamma \hat{I}(t)\big), \\
r_R(t) &= \frac{d\hat{R}}{dt} - \gamma \hat{I}(t).
\end{aligned}
$$

## 4. Physics Residual Loss Function

For sampled time points \${t\_i}\_{i=1}^N\$, the physics residual loss is:

$$
L_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \Big( r_S(t_i)^2 + r_E(t_i)^2 + r_I(t_i)^2 + r_R(t_i)^2 \Big).
$$

## 5. Total Loss

The complete PIKAN loss function can be written as:

$$
L = L_{\text{data}} + \lambda_{\text{phys}} L_{\text{physics}} + \lambda_{\text{cons}} L_{\text{conservation}},
$$

where the conservation constraint is:

$$
L_{\text{conservation}} = \frac{1}{N} \sum_{i=1}^N \Big( \hat{S}(t_i)+\hat{E}(t_i)+\hat{I}(t_i)+\hat{R}(t_i)-1 \Big)^2.
$$

