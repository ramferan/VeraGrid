
    def build_initial_lag_variables(self, x0:np.ndarray, dx0:np.ndarray, h) -> np.ndarray:
        if len(self._lag_vars) == 0:
            return

        x_lag = np.zeros(len(self._lag_vars), dtype=np.float64)

        lag_registry = self._lag_vars[0]._registry
        diff_registry = self._diff_vars[0]._absolute_registry

        max_order = max(var.diff_order for var in self._diff_vars)
        max_order = max(2, max_order)
        filtered_lag_dict = { key: value for key, value in lag_registry.items() if key[1] <= max_order }
        sorted_lag_dict = sorted(filtered_lag_dict.items(), key=lambda item: (item[0][0], item[0][1]))

        for key, lag_var in sorted_lag_dict:
            base_var_uid, lag = key
            uid = lag_var.uid 

            if lag == 0 or base_var_uid not in self.uid2idx_vars or uid not in self._lag_vars:
                continue

            idx = self.uid2idx_lag[uid] 
            x0_uid = self.uid2idx_vars[base_var_uid]

            # Collect previous dx0 and x_lag values for this lag_var
            dx0_slice = np.zeros(lag_var.lag)
            x_lag_last = 0

            for (prev_uid, prev_lag), prev_var in lag_registry.items():
                if prev_uid == base_var_uid and prev_lag <= lag and prev_lag !=0: 
                    try:
                        prev_diff     = diff_registry[base_var_uid, prev_lag]
                        prev_idx_diff = self.uid2idx_diff[prev_diff.uid]
                        dx0_slice[prev_lag-1] = dx0[prev_idx_diff]
                    
                    except:
                        if (base_var_uid, 1) in diff_registry and diff_registry[base_var_uid, 1] in self._diff_vars:
                            prev_diff     = diff_registry[base_var_uid, 1]
                            prev_idx_diff = self.uid2idx_diff[prev_diff.uid]
                            dx0_slice[prev_lag-1] = dx0[prev_idx_diff]
                        else:
                            dx0_slice[prev_lag-1] = 0


                    

            lag_i = lag_var.populate_initial_lag(x0[x0_uid], dx0_slice, x_lag_last, self.dt, h)
            if isinstance(lag_i, Expr):
                x_lag[idx] = lag_i.eval(dt = h)
            else:
                x_lag[idx] = lag_i
            _ = 0
        return x_lag
    
    def build_initial_guess(self, x0:np.ndarray, dx0:np.ndarray, h) -> np.ndarray:
        res = x0.copy() 
        for diff_var in self._diff_vars:
            if diff_var.diff_order > 1:
                continue 
            uid = diff_var.base_var.uid
            idx = self.uid2idx_vars[uid]
            diff_idx = self.uid2idx_diff[diff_var.uid]
            res[idx] += h*dx0[diff_idx]
        return res

    def simulate(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            dx0: np.ndarray,
            params0: np.ndarray,
            events_list: RmsEvents,
            method: Literal["rk4", "euler", "implicit_euler"] = "rk4",
            newton_tol: float = 1e-8,
            newton_max_iter: int = 1000,
            verbose =False

    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param events_list:
        :param params0:
        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        lag0 = self.build_initial_lag_variables(x0, dx0, h)
        x0   = self.build_initial_guess(x0, dx0, h)
        params_matrix = self.build_params_matrix(int(np.ceil((t_end - t0) / h)), params0, events_list)
        if method == "euler":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="euler")
        if method == "rk4":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="rk4")
        if method == "implicit_euler":
            return self._simulate_implicit_euler(
                t0, t_end, h, x0, dx0, lag0, params0, params_matrix,
                tol=newton_tol, max_iter=newton_max_iter, verbose = verbose,
            )
        raise ValueError(f"Unknown method '{method}'")

    def _simulate_implicit_euler(self, t0, t_end, h, x0, dx0, lag0, params0: np.ndarray, diff_params_matrix, tol=1e-8,
                                 max_iter=1000, verbose =False):
        """
        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :params_matrix:
        :param tol:
        :param max_iter:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_vars))
        speed_up = 1.0
        self.y = y
        self.t = t 
        params_current = params0
        diff_params_matrix = diff_params_matrix
        t[0] = t0
        y[0] = x0.copy()
        dx = dx0.copy()
        lag = np.asarray(lag0, dtype=np.float64)
        for step_idx in range(steps):
            self.step_idx = step_idx
            params_previous = params_current.copy()
            params_current += diff_params_matrix[step_idx, :].toarray().ravel()
            discontinuity = np.linalg.norm(params_current - params_previous, np.inf) > 1e-10
            xn = y[step_idx]
            x_new = xn.copy()  # initial guess
            converged = False
            n_iter = 0
            lambda_reg = 1e-6  # small regularization factor
            max_reg_tries = 1e6  # limit how much regularization is added
            reg_attempts = 0

            #We compute dx for the next step
            dx = self.compute_dx(x_new, lag, h)
            if step_idx == 0:
                max_iter = 100*max_iter
                tol = 1e4*tol
            elif step_idx == 1:
                max_iter = int(max_iter/100)
                tol = tol*1000     
            #print(f'dx is {dx}')

            while not converged and n_iter < max_iter:
                
                if discontinuity:
                    tol = 1e-2
                    speed_up = 100.0
                    max_iter = 5e5
                    print(f'dx is {dx}')
                    print(f'lag is {lag}')
                    #lag = self.build_initial_lag_variables(x_new, dx, h)
                    dx = self.compute_dx(x_new, lag, h)
                    #print(f'discontinuity at t = {t[step_idx]}, lag reset to {lag}')
                elif step_idx > 1:
                    tol = 1e-2
                    speed_up = 1.0
                    max_iter = 1e4

                xn_lags = np.r_[xn, lag]
                if step_idx < 0:
                    xnew_lags = np.r_[(x_new + xn)/2, lag]
                else:
                    xnew_lags = np.r_[x_new, lag]

                params_current = np.asarray(params_current, dtype=np.float64)
                if verbose:
                    print(f'[Run] solving system in iter {n_iter} and step {step_idx}')
                    print(f'With x_new = {xnew_lags} and xn is {xn_lags} and params {params_current}')
                    
                rhs = self.rhs_implicit(xnew_lags, xn_lags, params_current, step_idx + 1, h)
                Jf = self.jacobian_implicit(xnew_lags, params_current, h)  # sparse matrix

                if verbose:
                    print(f'RHS is {rhs} for x_new = {xnew_lags} and xn is {xn_lags} and params {params_current}')
                    #print(f'Jacobian is {Jf}')


                residual = np.linalg.norm(rhs, np.inf)
                converged = residual < tol

                if step_idx == 0:
                    old_lag = lag
                    lag = self.build_initial_lag_variables(x_new, dx0, h)
                    lag = 0.8*old_lag + 0.2*lag
                    if verbose:
                        print(f'Lag Change is {old_lag - lag}')
                        #print(f'self._lag_vars is {self._lag_vars}')
                        print(f'Lag is {lag}')
                        print(f'x_new of u is {x_new[19:21]}')
                    if converged:
                        print("System well initialized.")
                    else:
                        print(f"System bad initialized. DAE resiudal is {residual}.")

                if converged:
                    break
                
                Jf = self.jacobian_implicit(xnew_lags, params_current, h)  # sparse matrix
                solved = False

                while not solved and reg_attempts <= max_reg_tries:
                    try:
                        delta = sp.linalg.spsolve(Jf, -rhs)
                        solved = True
                    except:
                        print(f'[Run] lsqr try {step_idx} with iter {n_iter} and reg_attempts {reg_attempts}')
                        delta = sp.linalg.lsmr(Jf, -rhs, atol=1e-8, btol=1e-8, maxiter=1000)[0]
                        
                        if step_idx == 2500 and reg_attempts > 1000:
                            speed_up = 100.0
                        else:
                            speed_up = 1.0 
                        delta[19:21] = 1.0*delta[19:21]
                        print(f'Delta is {delta} and reg_attempts {reg_attempts} and speed_up {speed_up} and residual {residual}')
                        solved = True
                    reg_attempts += 1

                if not solved:
                    raise RuntimeError("Failed to solve linear system even with regularization.")

                x_new += delta
                n_iter += 1

            if converged:
                print(f'converged {converged} and n_iter {n_iter} and iter {n_iter}')
                if verbose:
                    print(f'delta is {delta} and x_new {x_new}')
                    print(f'lag is {lag}')
                if discontinuity:
                    _=0
                    y[step_idx + 1] = x_new 
                else:
                    y[step_idx + 1] = x_new
                t[step_idx + 1] = t[step_idx] + h

                for i, lag_var in enumerate(self._lag_vars):
                    if step_idx >= (lag_var.lag-1):
                        uid = lag_var.base_var.uid
                        idx = self.uid2idx_vars[uid]
                        lag[i] = y[step_idx + 1 - (lag_var.lag-1), idx]
                    else:
                        lag_name = lag_var.base_var.name + '_lag_' + str(lag_var.lag-1)
                        next_lag_var = LagVar.get_or_create(lag_name, base_var= lag_var.base_var, lag = lag_var.lag-1)
                        uid = next_lag_var.uid
                        idx = self.uid2idx_lag[uid]
                        lag[i] = lag[idx] 
            else:
                print(f"Failed to converge at step {step_idx}")
                print(f'Jacobian is {Jf}')
                break

        return t, y
    
    
    
    def compute_dx(self, x:np.ndarray, lag: np.ndarray, h: float) -> np.ndarray:
        """
        Compute the numerical derivative (dx) for all differential variables 
        using symbolic approximation expressions and lagged variables.
    
        Parameters
        ----------
        y : np.ndarray
            State variable trajectory. `y[-1, :]` corresponds to the most recent 
            values of the system variables.
        lag : np.ndarray
            Array containing lagged values of variables (delayed states).
        h : float
            Time step (dt) used in the approximation.
    
        Returns
        -------
        np.ndarray
            Array with computed derivatives for each differential variable, 
            indexed consistently with `self._diff_vars`.
        """
        res = np.zeros( len(self._diff_vars), dtype=np.float64)
        for diff_var in self._diff_vars:
            uid = diff_var.uid
            idx = self.uid2idx_diff[uid]
            dx_expression, lag_number = diff_var.approximation_expr(self.dt)

            #We substitute the origin variable and dt
            subs = {diff_var.origin_var: Const(x[self.uid2idx_vars[diff_var.origin_var.uid]])}
            subs[self.dt] = Const(h)

            #We substitute the lag variables
            i = 1
            lag_in_expression = True
            while lag_in_expression or i<=2:
                lag_i = LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(i),   
                                            base_var=diff_var.origin_var, lag = i)
                dx_expression = dx_expression.subs({self.dt:Const(h)})
                deriv = dx_expression.diff(lag_i)
                if getattr(deriv, 'value', 1) == 0:
                    if i>2:
                        lag_in_expression = False
                        i = i+1
                        break
                    i += 1
                    continue
                lag_idx = self.uid2idx_lag[lag_i.uid]

                subs[lag_i] = Const(lag[lag_idx])
                i += 1

            deriv_value = dx_expression.subs(subs)
            deriv_value = deriv_value.eval()
            res[idx] = deriv_value

        return res
    
    def test_equations(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            dx0: np.ndarray,
            params0: np.ndarray,
            events_list: RmsEvents,
            method: Literal["rk4", "euler", "implicit_euler"] = "rk4",
            newton_tol: float = 1e-8,
            newton_max_iter: int = 1000,

    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param events_list:
        :param params0:
        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        lag0 = self.build_initial_lag_variables(x0, dx0, h)
        x0   = self.build_initial_guess(x0, dx0, h)

        params_matrix = self.build_params_matrix(int(np.ceil((t_end - t0) / h)), params0, events_list)
        params_current = params0
        params_current += params_matrix[0, :].toarray().ravel()
        xn_lags = np.concatenate((x0, lag0))
        print(f' xn is {xn_lags}, params is {params_current}')
        rhs = self.rhs_implicit(xn_lags, xn_lags, params_current, 0, h)
        print(f"rhs is {rhs}")
        Jf = self.jacobian_implicit(xn_lags, params_current, h)  # sparse matrix
        print(f"Jf is {Jf}")

        return
        
        
def _get_jacobian:
    if getattr(deriv, 'value', 1) != 0 and diff_var.origin_var.uid == var.uid:
    name_diff = 'diff_' + diff_var.name
    diff_2_var = DiffVar.get_or_create(name = name_diff, base_var=diff_var)
    dx2_dt2, lags1 = diff_2_var.approximation_expr(dt=dt)
    dx_dt, lags2   = diff_var.approximation_expr(dt=dt)
    #d_expression = dx_dt*d_expression + dx2_dt2*eq.diff(diff_var).simplify()
    #To use the expression above we would need to multiply every value by dxdt and change the RHS
    #TO DO: see if its really needed to ensure multilinearity
    d_expression += (dx2_dt2/dx_dt)*eq.diff(diff_var).simpli
    set_lags = set( LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(lag), 
                            base_var = diff_var.origin_var, lag = lag) for lag in range(1, max(lags1, lags2)))
    new_lags = set_lags - self._lag_vars
    #We add the lag to the index
    self._lag_vars_set.update(new_lags)
    i = len(self.uid2idx_vars)
    l = len(self.uid2idx_lag)
    for v in new_lags:  # deterministic
        uid2sym_vars[v.uid] = f"vars[{i}]"
        self.uid2idx_vars[v.uid] = i
        self.uid2idx_lag[v.uid] = l
        self._lag_vars.append(v)
        i += 1
        l += 1
    
    k = len(self.uid2idx_diff)
    if diff_2_var not in self._diff_vars:
        self.uid2idx_diff[diff_2_var.uid] = k
        k += 1