import numpy as np
from .constraints import Constraints
import time 

class Cost:

  def __init__(self, params):

    self.params = params
    self.soft_constraints = Constraints(params)

    # load parameters
    self.T = params['T']  # Planning Time Horizon
    self.N = params['N']  # number of planning steps
    self.dt = self.T / (self.N - 1)  # time step for each planning step
    self.v_max = params['v_max']  # max velocity

    # cost
    self.w_vel = params['w_vel']
    self.w_contour = params['w_contour']
    self.w_theta = params['w_theta']
    self.w_accel = params['w_accel']
    self.w_delta = params['w_delta']
    self.wheelbase = params['wheelbase']

    self.track_offset = params['track_offset']

    self.W_state = np.array([[self.w_contour, 0], [0, self.w_vel]])
    self.W_control = np.array([[self.w_accel, 0], [0, self.w_delta]])

    # useful constants
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

  #* New stuff.
  def update_obs(self, frs_list):
    self.soft_constraints.update_obs(frs_list)

  def get_cost(self, states, controls, leader_states, closest_pt, slope, theta):
    """
    Calculates the cost given planned states and controls.

    Args:
        states (np.ndarray): 4xN array of planned trajectory.
        controls (np.ndarray): 2xN array of planned control.
        closest_pt (np.ndarray): 2xN array of each state's closest point [x,y]
            on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest
            points.
        theta (np.ndarray): 1xN array of the progress at each state.

    Returns:
        np.ndarray: costs.
    """
    
    c_control = np.einsum(
        'an, an->n', controls,
        np.einsum('ab, bn->an', self.W_control, controls)
    )

    c_control[-1] = 0
    
    # constraints
    c_constraint = self.soft_constraints.get_cost(
        states, controls, closest_pt, slope
    )    

    
    ref_states = leader_states #[x,y,v,psi,t]
    
    error = states - ref_states 
    
    J = 0.7*(error[0,:]**2 + error[1,:]**2) + 0.2*error[2,:]**2 + 0.1*error[3,:]
    J = np.sum(J + c_constraint + c_control)
    #J = np.linalg.norm(error)**2
    return J

  def get_derivatives(self, states, controls, leader_waypoints, closest_pt, slope):
    '''
    Calculate Jacobian and Hessian of the cost function
        states: 4xN array of planned trajectory
        controls: 2xN array of planned control
        closest_pt: 2xN array of each state's closest point [x,y] on the track
        slope: 1xN array of track's slopes (rad) at closest points
    '''

    c_x_cons, c_xx_cons, c_u_cons, c_uu_cons, c_ux_cons = (
        self.soft_constraints.get_derivatives(
            states, controls, closest_pt, slope
        )
    )

    c_x_cost, c_xx_cost = self._get_cost_state_derivative(
        states, leader_waypoints
    )

    c_u_cost, c_uu_cost = self._get_cost_control_derivative(controls)

    q = c_x_cons + c_x_cost
    Q = c_xx_cons + c_xx_cost

    r = c_u_cost + c_u_cons
    R = c_uu_cost + c_uu_cons

    S = c_ux_cons

    return q, Q, r, R, S

  def _get_cost_state_derivative(self, states, leader_waypoints):
    '''
    Calculate Jacobian and Hessian of the cost function with respect to state
        states: 4xN array of planned trajectory
        closest_pt: 2xN array of each state's closest point [x,y] on the track
        slope: 1xN array of track's slopes (rad) at closest points
    '''
    ref_states = leader_waypoints

    error = states - ref_states 
   
    # shape [4xN]

    J = 0.7*(error[0]**2 + error[1]**2) + 0.2*error[2]**2 + 0.1*error[3]


    c_x = 2*error
    c_x[0,:] = 0.7*c_x[0,:]
    c_x[1,:] = 0.7*c_x[1,:]
    c_x[2,:] = 0.2*c_x[2,:]
    c_x[3,:] = 0.1*c_x[3,:]
    

    c_xx = np.zeros([4,4,self.N])
    for i in range(0,self.N):
        c_xx[:,:,i] = 2*np.eye(4)
        c_xx[0,0,i] = 0.7*c_xx[0,0,i]
        c_xx[1,1,i] = 0.7*c_xx[1,1,i]
        c_xx[2,2,i] = 0.2*c_xx[2,2,i]
        c_xx[3,3,i] = 0.1*c_xx[3,3,i]


    return c_x, c_xx

  def _get_cost_control_derivative(self, controls):
    '''
    Calculate Jacobian and Hessian of the cost function w.r.t the control
        controls: 2xN array of planned control
    '''
    c_u = 2 * np.einsum('ab, bn->an', self.W_control, controls)
    c_uu = 2 * np.repeat(self.W_control[:, :, np.newaxis], self.N, axis=2)
    return c_u, c_uu
