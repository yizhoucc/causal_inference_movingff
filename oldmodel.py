
class CircularFireflyEnv(gym.Env):

    def __init__(self):

        # Constants for the task
        self.tobs = 0.3  # Observation time in seconds
        self.T = 1.5  # Action time in seconds
        self.delta_t = 0.05
        self.n_steps_obs = int(self.tobs / self.delta_t) + \
            1  # Number of steps during obs

        # Noise parameters
        self.sigma_f = 0.3  # Noise in firefly v observation
        self.sigma_w = 0.3  # Noise in self-motion v observation

        # Action and observation space
        self.action_space = spaces.Box(
            low=-60, high=60, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        '''
            reset the task variables
            x0, inital state
            f0, the angle in degree
            w, angular speed
            v, radial speed
        '''
        # Initial positions and velocities
        self.x0 = 0
        self.f0 = np.random.uniform(-180, 180)
        self.w = np.random.uniform(-50, 50)  # Subject's angular speed
        if abs(self.w) > 25:
            self.w = 0
            print('w is 0')
        self.v = np.random.uniform(-50, 50)  # Firefly's velocity
        if abs(self.v) > 25:
            self.v = 0
            print('v is 0')
        self.current_time = 0

        self.A = np.array([[1, 1], [0, 0]])
        self.Q = np.array([[0.1]])  # self estimated process noise
        # self estimated obs noise. these 0.1 are tmp for now
        self.R = np.array([[0.1]])

        self.obs_phase()
        self.init_belief()

        return self.belief

    def step(self, a):

        # state steps
        self.current_time += self.delta_t
        s = self.update_state(s)
        s = self.apply_action(s, a)

        # belief steps
        o = self.get_vel()
        b = self.belief_step(self.belief, self.P, o, a)

        # finalizing
        self.s = s
        self.b = b

        # TODO Compute reward
        distance_to_firefly = np.abs(s[0])
        # Negative reward based on distance
        reward = (distance_to_firefly < 10)

        done = reward | self.current_time >= self.T/self.delta_t
        info = {}

        return self.belief, reward, done, info

    def update_state(self, s):
        ''' increase dist by v*dt'''
        next_s = s.copy()
        next_s[0] += next_s[1]*self.delta_t
        return next_s

    def apply_action(self, s, a):
        '''replace the velocity in state with current control'''
        next_s = s.copy()
        next_s[1] = a
        return next_s

    def get_vel(self,):
        '''get velocity during navigation'''
        noise = np.random.normal(
            0, 0.1 / np.sqrt(self.delta_t))  # 0.1 tmp for now
        return self.s[1]+noise

    def belief_step(self, previous_b, previous_P, o, a):
        '''kalman filter belief udpate in action phase'''
        I = np.eye(2)
        H = np.tensor([[0., 1.]])
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b, a)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q

        error = o - H@predicted_b
        S = H@(predicted_P)@(H.t()) + self.R
        K = predicted_P@(H.t())@(np.inverse(S))

        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)

        return b, P

    def obs_phase(self,):
        '''
            obs phase. collet the obs into a list, saved in class attr
            input: none
            return: none
        '''
        # Generate noisy observations
        self.observations = []
        for i in range(self.n_steps_obs):
            v_obs = np.random.normal(
                self.v, self.sigma_f / np.sqrt(self.delta_t))
            w_obs = np.random.normal(
                self.w, self.sigma_w / np.sqrt(self.delta_t))
            # d_obs = v_obs-w_obs # this is what subject actually obs. the diff
            self.observations.append([w_obs, v_obs])

    def init_belief(self,):
        '''causual inference to get belief of the firefly in obs phase'''
        self.belief = None
        move = self.is_move()
        ws = np.mean([a[0] for a in self.observations])
        vs = np.mean([a[1] for a in self.observations])
        if move:
            move_vel = vs-ws  # the difference in self and target velocities.
        else:
            move_vel = vs
        move_dist = move_vel*self.n_steps_obs*self.delta_t
        # cur-target state distant, cur-target state vel
        self.belief = np.array([move_dist, move_vel])
        self.P = np.eye(2) * 1e-8

    def is_move(self,):
        '''decide if the firefly is a moving one'''
        B = [a[1] for a in self.observations]
        # get the pdf of observed firefly velocities
        mu, sigma = calculate_mu_sigma(B)
        prob_density_0 = probability_of_zero(mu, sigma)
        return prob_density_0 < 0.5

    def render(self, mode='human'):
        '''TODO for visualizations '''
        print(
            f'Time: {self.current_time}, Position: {self.state[-2]}, Firefly: {self.state[-1]}')
