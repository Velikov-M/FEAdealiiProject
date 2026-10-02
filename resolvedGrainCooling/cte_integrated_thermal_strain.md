# Secant vs instantaneous CTE: proposal to make the thermal strain primary

**Status: PARKED (2026-10-02).** The model keeps its current secant formulation for now. This note
records the problem, how large it is, and the change proposed for when it is picked up.

## 1. The problem

The model computes the thermal strain as ([main.cpp:1557](main.cpp#L1557), assembly near
[main.cpp:2026](main.cpp#L2026)):

$$\varepsilon^\theta(T) = \alpha(T)\,(T - T_0), \qquad T_0 = \texttt{InitialTemperature} = 2000\ \mathrm{K}$$

so its $\alpha$ is a **secant (mean) CTE measured from the stress-free temperature $T_0$**.

Literature CTE curves are **instantaneous**, $\alpha_{\mathrm{inst}} = d\varepsilon/dT$:

- Marsden et al. 2018, eq. (2.6): $\alpha_c = \partial e_{zz}/\partial T$.
- The compilation figure behind the current `alpha11_0 = 1.2e-6` and `alpha33_0 = 32e-6` (Steward
  et al.) shows $\alpha_c \to 0$ as $T \to 0$ and a negative $\alpha_a$ between 300 and 600 K.
  Neither is possible for a mean taken from 2000 K.

The two are related by

$$\alpha_{\mathrm{sec}}(T) = \frac{1}{T - T_0}\int_{T_0}^{T}\alpha_{\mathrm{inst}}(T')\,dT'$$

Feeding $\alpha_{\mathrm{inst}}$ into the secant slot is wrong in general. The entropy argument in
[thermodynamic_potential.md §7](thermodynamic_potential.md) says the same: the coefficient conjugate
to stress is $d\varepsilon^\theta/dT$.

## 2. How big it is

The table uses the by-eye Steward points and gives the mismatch $\alpha_c - \alpha_a$, which drives
the stresses (units 1e-6/K):

| T [K] | code uses (32 - 1.2) | exact secant from 2000 K | error |
|---|---|---|---|
| 1980 | 30.80 | 31.75 | -3.0% |
| 1800 | 30.80 | 31.37 | -1.8% |
| 1500 | 30.80 | 30.73 | +0.2% |
| 1000 | 30.80 | 29.66 | +3.8% |
| 300 | 30.80 | 28.52 | +8.0% |

- **For runs done so far (2000 -> 1800 K) the error is 2-3%,** about as precise as reading a figure.
  No result is invalidated.
- **It becomes a real error when cooling far below 1800 K, or once $\alpha$ is given a T-dependence.**
  The basal plane shows it most clearly: at 300 K, $\alpha_{a,\mathrm{inst}} = -1.35$ but
  $\alpha_{a,\mathrm{sec}} = +0.58$. Putting the instantaneous curve into $\alpha(T)\,\Delta T$ gets
  the sign of the accumulated basal strain wrong.

A workaround already exists without changing code: give the model the **secant** curve. The
`materialData/graphite_cte_CCM_Marsden2018/` fits provide `secant_functionOfTemperature` for exactly
this purpose. The cost is that the secant is tied to $T_0$, so changing `InitialTemperature` requires
regenerating it.

## 3. Proposal: make $\varepsilon^\theta$ the primary quantity

Replace "$\alpha$ times $\Delta T$" with the integrated strain, in the local (crystal) frame:

$$\varepsilon^\theta_{\mathrm{loc}}(T,\omega) = \mathrm{diag}\big(\varepsilon_a(T),\ \varepsilon_a(T),\ (1-\omega)\,\varepsilon_c(T)\big), \qquad \varepsilon_i(T) = \int_{T_0}^{T}\alpha_{i,\mathrm{inst}}(T')\,dT'$$

- The input becomes the instantaneous $\alpha_{i,\mathrm{inst}}(T)$, exactly as published. There is no
  conversion step, and $T_0$ can change freely.
- $\varepsilon_i(T)$ has a closed form for both fit families already in use. Polynomial fits
  integrate term by term. The Einstein fit integrates as
  $\int A\,E(\theta/T)\,dT = A\,\theta/(e^{\theta/T}-1)$.
  A user-supplied FunctionParser $\alpha_{\mathrm{inst}}$ could instead be integrated numerically
  (Gauss, a few points per step). Either way the cost is negligible, because T is uniform in space:
  one evaluation per temperature step.
- **Code impact is local.** `CteTensor::getcteTensor(omega, T)` becomes
  `getThermalStrain(omega, T)`, which returns the strain itself. Every `cte * deltaT` product becomes
  that strain: the stress in `getLocalStressTensor`, the assembly RHS, `computeAverageStress`, and the
  per-grain cache (`grainCteZeroDamage` and `grainCacheAlpha33` become strains). The rank-one damage
  identity survives unchanged:
  $\varepsilon^\theta_{\mathrm{glob}}(\omega) = \varepsilon^\theta_{\mathrm{glob}}(0) - \omega\,\varepsilon_c\,n\otimes n$.
- **Output.** The effective CTE $\bar\alpha$ reported in the history CSV is currently
  $\bar E/\Delta T$, also a secant. Keep it, but label it as secant, and optionally add the
  instantaneous version $\Delta\bar E/\Delta T$ per step.
- **Verification.** With constant $\alpha_{\mathrm{inst}}$ the new code must reproduce the current
  results bit for bit. With T-dependent $\alpha$ it must match the old code fed the secant curve from
  `cte_fits.json`.

## 4. Design question to settle first: total or incremental damage scaling

Currently damage scales the **total** c-axis thermal strain: $(1-\omega)\,\varepsilon_c(T)$. Damage
gained at 1850 K therefore also cancels the contraction accumulated between 2000 and 1850 K. The
alternative scales only what follows:

$$d\varepsilon^\theta_{33} = (1-\omega)\,\alpha_{c,\mathrm{inst}}(T)\,dT$$

| | total (current) | incremental |
|---|---|---|
| state | $\psi(\varepsilon, T, \omega)$: a state function | $\varepsilon^\theta$ becomes a history variable |
| thermodynamics | the potential, driving force $Y$ and the $Y > 0$ check of `thermodynamic_potential.md` hold as derived | $Y$ loses the $-\hat\alpha_{33}\,\Delta T\,\sigma_{33}$ term (term (b), currently 2.1x term (a)); needs re-deriving |
| physics | cracks retroactively accommodate the whole contraction | cracks only accommodate contraction after they open: arguably closer to Mrozowski crack opening |
| heating / healing | reversible by construction | path dependent; needs its own rule on reheating |

**Recommendation: keep the total form** and switch only the parameterisation (sections 3 and 4 are
independent). Revisit if the heating branch needs path dependence.

## 5. Second-order caveat: which length the strain is measured from

$\alpha = \partial e/\partial T$ in Marsden is measured relative to the current lattice parameter, so
integrating it gives a logarithmic strain. The model is small-strain. Over 2000 -> 1800 K the
c-axis strain is about 0.7% and the difference is negligible. From 2000 down to 300 K it is about 6%,
and log vs engineering strain differ by about 3% of that. This is below the spread between data sets
in the figure (Morgan vs CCM differ by about 8%), so it is not worth handling unless the model goes
finite-strain.
