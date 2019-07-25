"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        delta_t = 1.
        self.state = np.matrix([[init_x], [init_y], [0.], [0.]])  # state
        self.cov = np.matrix(np.eye(4, dtype=float))
        self.Dt = np.matrix([[1., 0., delta_t, 0.],
                             [0., 1., 0., delta_t],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
        self.Mt = np.matrix([[1., 0., 0., 0.],
                            [0., 1., 0., 0.]])
        self.sig_d = Q
        self.sig_m = R

    def predict(self):
        self.state = self.Dt * self.state
        self.cov = self.Dt * self.cov * self.Dt.T + self.sig_d

    def correct(self, meas_x, meas_y):
        inv = self.Mt * self.cov * self.Mt.T + self.sig_m
        Kt = self.cov * self.Mt.T * np.linalg.inv(inv)
        Yt = np.array([[meas_x], [meas_y]])
        self.state = self.state + Kt * (Yt - self.Mt * self.state)
        self.cov = (np.identity(4) - Kt * self.Mt) * self.cov

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        self.particles = np.zeros((self.num_particles, 2), dtype=int)  # Initialize your particles array. Read the docstring.
        x, y, w, h = self.template_rect['x'], self.template_rect['y'], self.template_rect['w'], self.template_rect['h']
        self.particles[:, 0] = x+w//2
        self.particles[:, 1] = y+h//2
        #self.update_particle()
        self.weights = np.ones(self.num_particles)/self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        n, m = template.shape
        mse = np.sum((template - frame_cutout)**2) / float(m*n)
        return np.exp(-mse/(2. * self.sigma_exp**2))

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """

        ind = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        return self.particles[ind]

    def update_particle(self):
        self.particles[:, 0] += (np.random.normal(0, self.sigma_dyn, self.num_particles)).astype(int)
        self.particles[:, 1] += (np.random.normal(0, self.sigma_dyn, self.num_particles)).astype(int)

    def cutout(self, template, p):
        h, w = template.shape
        cutout = np.zeros((h, w))
        frame_border = np.ceil([[p[0] - w/2, p[0] + w/2], [p[1] - h/2, p[1] + h/2]]).astype(int)
        cutout_border = [[0, w], [0, h]]

        for i in range(2):
            if frame_border[i][0] < 0:
                cutout_border[i][0] = -1*frame_border[i][0]
                frame_border[i][0] = 0
            if frame_border[i][1] > self.frame.shape[1-i]:
                cutout_border[i][1] = self.frame.shape[1-i] - frame_border[i][0]
                frame_border[i][1] = self.frame.shape[1-i]
            if (frame_border[i][0] > self.frame.shape[1-i]) or (frame_border[i][1] < 0):
                return cutout
        try:
            cutout[cutout_border[1][0]:cutout_border[1][1], cutout_border[0][0]:cutout_border[0][1]] = \
                self.frame[frame_border[1][0]:frame_border[1][1], frame_border[0][0]:frame_border[0][1]]
        except:
            return cutout

        return cutout

    def calc_weights(self):
        for i in range(self.num_particles):
            self.weights[i] = self.get_error_metric(self.template, self.cutout(self.template, self.particles[i]))
        self.weights /= sum(self.weights)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        self.update_particle()
        self.calc_weights()
        self.particles = self.resample_particles()

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        x_weighted_mean, y_weighted_mean = int(x_weighted_mean), int(y_weighted_mean)
        # Complete the rest of the code as instructed.
        for particle in self.particles:
            cv2.circle(frame_in, (int(particle[0]), int(particle[1])), 1, (0, 0, 255), thickness=-1)

        h, w = self.template.shape
        cv2.rectangle(frame_in, (x_weighted_mean-w//2, y_weighted_mean-h//2),
                      (x_weighted_mean+w//2, y_weighted_mean+h//2), (0, 255, 0), thickness=1)

        dist = np.linalg.norm(self.particles[:, :2] - [x_weighted_mean, y_weighted_mean], axis=1)
        r = np.average(dist, axis=0, weights=self.weights).astype(int)
        cv2.circle(frame_in, (x_weighted_mean, y_weighted_mean), r, (255, 0, 0), thickness=1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """

        super(AppearanceModelPF, self).process(frame)

        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        x_weighted_mean, y_weighted_mean = int(x_weighted_mean), int(y_weighted_mean)

        best = self.cutout(self.template, [x_weighted_mean, y_weighted_mean])
        self.template = self.alpha * best + (1 - self.alpha) * self.template
        self.template = cv2.normalize(self.template, None, 0, 255, cv2.NORM_MINMAX)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.sigma_size = kwargs.get('sigma_size')
        self.particles = np.zeros((self.num_particles, 3))
        x, y, w, h = self.template_rect['x'], self.template_rect['y'], self.template_rect['w'], self.template_rect['h']
        self.particles[:, 0] = int(x + w // 2)
        self.particles[:, 1] = int(y + h // 2)
        self.particles[:, 2] = 1
        #self.update_particle()
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def update_particle(self):
        super(MDParticleFilter, self).update_particle()
        self.particles[:, 2] = np.random.normal(1, self.sigma_size, self.num_particles)

    def calc_weights(self):
        for i in range(self.num_particles):
            template = cv2.resize(self.template, None, fx=self.particles[i][2], fy=self.particles[i][2], interpolation=cv2.INTER_CUBIC)
            self.weights[i] = self.get_error_metric(template, self.cutout(template, self.particles[i]))
        self.weights /= sum(self.weights)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        super(MDParticleFilter, self).process(frame)

        size_weighted_mean = 0
        for i in range(self.num_particles):
            size_weighted_mean += self.particles[i, 2] * self.weights[i]

        self.template = cv2.resize(self.template, None, fx=size_weighted_mean, fy=size_weighted_mean, interpolation=cv2.INTER_CUBIC)