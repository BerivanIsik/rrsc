# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code to obtain parameters of various miracle methods."""

import numpy as np
import scipy.special as sc

from utilities.optimize_unbias import *


def get_privunit_densities(d, gamma, p):
  """Compute the constants that the conditional density is proportional to.
  The conditional density of z (i.e., the output of the privunit) given x (i.e.,
  the input to the privunit) is proportional to c1 if the inner product between
  x and z is more than gamma and is proportional to c2 otherwise.
  Args:
    d: The number of dimensions.
    gamma: The best gamma.
    p : The probability with which an unit vector is sampled from the shaded
    spherical cap associated with the input (see the original paper).
  Returns:
    c1: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is more than gamma.
    c2: The factor that the conditional density of z given x is proportional to
    if the inner product between x and z is less than gamma.
  """
  c1 = 2 * p / (sc.betainc((d - 1) / 2, 1 / 2, (1 - gamma**2)))
  c2 = 2 * (1 - p) / (2 - sc.betainc((d - 1) / 2, 1 / 2, (1 - gamma**2)))
  return c1, c2

def get_parameters_unbiased_approx_miracle(epsilon_target, d,
                                           approx_number_candidates,
                                           number_candidates, budget, delta):
  """Get privunit parameters for miracle with approxmiate-DP guarantees.
  The approximate DP guarantee of miracle is as follows: If a mechanism is
  epsilon-DP, simulating it with MIRACLE (with N candidates) gives a mechanism
  which is (epsilon + log(1 + t) - log(1 - t), delta)-DP where
  delta = 2 * exp(-N * t^2 / (c_1 - c_2)^2). An important point to note here is
  that c_1 and c_2 are themselves functions of epsilon. So for given delta,
  number of candidates, and epsilon_target, the idea is to come up with the
  largest epsilon for which epsilon + log(1 + t) - log(1 - t) is smaller than
  epsilon_target.
  Args:
    epsilon_target: The privacy guarantee we desire.
    d: The number of dimensions.
    approx_number_candidates: Approximate number of candidates.
    number_candidates: The number of candidates.
    budget: The default budget splitting between the gamma and p parameters.
    delta: The delta in the differential privacy guarantee.
  Returns:
    c1: The larger constant that the privunit density is proportional to.
    c2: The smaller constant that the privunit density is proportional to.
    m: The inverse of the scalar norm that the decoder should use to get
    an unbiased estimator.
    gamma: The gamma parameter of privunit.
    epsilon_approx: The resulting epsilon that this version of miracle ends
    up using.
  """

  epsilon_search_space = np.linspace(0, epsilon_target, 200)
  epsilon_search_space = epsilon_search_space[:-1]
  epsilon_approx = 0
  # Find the largest epsilon for PrivUnit so that MIRACLE meets epsilon_target
  for epsilon in epsilon_search_space:
    gamma = find_best_gamma(d, budget * epsilon)
    p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
    c1, c2 = get_privunit_densities(d, gamma, p)
    t = np.abs(c1 - c2) * np.sqrt(
        (np.log(2 / delta)) / (2 * approx_number_candidates))
    if -1 < t < 1:
      if epsilon + np.log(1 + t) - np.log(1 - t) <= epsilon_target:
        epsilon_approx = epsilon
  gamma = find_best_gamma(d, budget * epsilon_approx)
  p = np.exp((1 - budget) * epsilon_approx) / (1 + np.exp(
      (1 - budget) * epsilon_approx))
  c1, c2 = get_privunit_densities(d, gamma, p)
  p_hat = get_unbiased_p_hat(number_candidates, c1, c2, p)
  m_hat = getm(d, gamma, p_hat)

  return c1, c2, m_hat, gamma, epsilon_approx


def get_parameters_unbiased_miracle(epsilon, d, number_candidates,
                                             budget):
  """Get privunit parameters for unbiased miracle."""
  # Get the optimized budget.
  gamma = find_best_gamma(d, budget * epsilon)
  p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
  c1, c2 = get_privunit_densities(d, gamma, p)
  p_tilde = get_unbiased_p_tilde(number_candidates, c1, c2, p,
                                                 epsilon)
  m_tilde = getm(d, gamma, p_tilde)

  return c1, c2, m_tilde, gamma

# Find best gamma for MMRC.
def find_best_gamma(d, eps):
  """This function finds the best gamma in an iterative fashion.
  Gamma is essentially the parameter in the privunit algorithm that specifies
  the distance from the equator (see figure 2 in the original paper linked
  above). The best gamma here refers to the one that achieves maximum accuracy.
  Gamma always adheres to (16a) or (16b) in the original paper (linked above).
  Args:
    d: Number of dimensions.
    eps: The privacy parameter epsilon.
  Returns:
    gamma: The best gamma.
  """
  gamma_a = (np.exp(eps) - 1) / (np.exp(eps) + 1) * np.sqrt(np.pi / (2 * d - 2))

  # Calculate an upper bound on gamma as the initialization step.
  gamma_b = min(np.exp(eps) / (6 * np.sqrt(d)), 1)
  while eps < 1 / 2 * np.log(
      d * 36) - (d - 1) / 2 * np.log(1 - gamma_b**2) + np.log(gamma_b):
    gamma_b = gamma_b / 1.01

  if gamma_b > np.sqrt(2 / d):
    gamma = max(gamma_b, gamma_a)
  else:
    gamma = gamma_a

  return gamma

def getm(d, gamma, p):
  """Get the parameter m (eq (15) in the paper) in the privunit mechanism."""
  alpha = (d - 1) / 2
  tau = (1 + gamma) / 2
  if d > 1000:
    # For large d, Stirling's formula is used to approximate eq (15).
    m = (d - 2) / (d - 1) * (1 - gamma**2)**alpha / (
        2 * np.sqrt(np.pi * (d - 3) / 2)) * (
            p / (1 - sc.betainc(alpha, alpha, tau)) -
            (1 - p) / sc.betainc(alpha, alpha, tau))
  else:
    # For small d, eq (15) is used directly
    m = ((1 - gamma**2)**alpha) * (
        (p / (sc.betainc(alpha, alpha, 1) - sc.betainc(alpha, alpha, tau))) -
        ((1 - p) / sc.betainc(alpha, alpha, tau))) / (
            (2**(d - 2)) * (d - 1) * sc.beta(alpha, alpha))
  return m

# Find optimized budget for MMRC.
def get_optimized_budget(epsilon, d):
  budget_space = np.linspace(0.01, 0.99, 99)
  m = np.zeros(len(budget_space))
  for step, budget in enumerate(budget_space):
    gamma = find_best_gamma(d, budget * epsilon)
    p = np.exp((1 - budget) * epsilon) / (1 + np.exp((1 - budget) * epsilon))
    m[step] = getm(d, gamma, p)
  return budget_space[np.argmax(m)]