import numpy as np
#---------------------------------------------------
#This document contains all functions needed for Maxwell_Solver.py
#All derivatives are using finite difference methods.
#---------------------------------------------------

class Functions():
    def __init__(self,delta,grid_size):
        #Import delta and grid_size from Maxwell_solver.py
        self.delta = delta
        self.grid_size = grid_size

    def gradient(self,field):
        grad_x = np.zeros((self.grid_size,self.grid_size,self.grid_size))
        grad_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        grad_x[1:-1,1:-1,1:-1] = (field[2::,1:-1,1:-1]-field[0:-2,1:-1,1:-1])/(2*self.delta)
        grad_y[1:-1,1:-1,1:-1] = (field[1:-1,2::,1:-1]-field[1:-1,0:-2,1:-1])/(2*self.delta)
        grad_z[1:-1,1:-1,1:-1] = (field[1:-1,1:-1,2::]-field[1:-1,1:-1,0:-2])/(2*self.delta)
        grad = np.array([grad_x,grad_y,grad_z])
        return grad

    def divergence(self,field):
        grad_x = self.gradient(field[0, :, :, :])
        grad_y = self.gradient(field[1, :, :, :])
        grad_z = self.gradient(field[2, :, :, :])
        componentDiv = grad_x[0]+grad_y[1]+grad_z[2]
        return componentDiv

    def laplac(self,field):
        componentLap = np.sum(self.derivativeo2(field),axis=0)
        return componentLap

    def derivativeo2(self,field):
        deriv_x = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
        deriv_y = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
        deriv_z = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))

        deriv_x[:,1:-1,1:-1,1:-1] = (field[:,2::,1:-1,1:-1]-2*field[:,1:-1,1:-1,1:-1]+field[:,0:-2,1:-1,1:-1])/self.delta**2
        deriv_y[:,1:-1,1:-1,1:-1] = (field[:,1:-1,2::,1:-1]-2*field[:,1:-1,1:-1,1:-1]+field[:,1:-1,0:-2,1:-1])/self.delta**2
        deriv_z[:,1:-1,1:-1,1:-1] = (field[:,1:-1,1:-1,2::]-2*field[:,1:-1,1:-1,1:-1]+field[:,1:-1,1:-1,0:-2])/self.delta**2
        derivo2 = np.array([deriv_x,deriv_y,deriv_z])
        return derivo2

    def grad_div(self, field):
        grad_div_x = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_div_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        grad_div_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        xx_A_x = (field[0,2::,1:-1,1:-1]-2*field[0,1:-1,1:-1,1:-1]+field[0,0:-2,1:-1,1:-1])/self.delta**2
        xy_A_y = (field[1,2::,2::,1:-1]-field[1,2::,0:-2,1:-1]-field[1,0:-2,2::,1:-1]+field[1,0:-2,0:-2,1:-1])
        xz_A_z = (field[2,2::,1:-1,2::]-field[2,2::,1:-1,0:-2]-field[2,0:-2,1:-1,2::]+field[2,0:-2,1:-1,0:-2])
        grad_div_x[1:-1,1:-1,1:-1] = self.delta**2*(xx_A_x+0.25*xy_A_y+0.25*xz_A_z)

        yx_A_x = (field[0,2::,2::,1:-1]-field[0,2::,0:-2,1:-1]-field[0,0:-2,2::,1:-1]+field[0,0:-2,0:-2,1:-1])
        yy_A_y = (field[1,1:-1,2::,1:-1]-2*field[1,1:-1,1:-1,1:-1]+field[1,1:-1,0:-2,1:-1])
        yz_A_z = (field[2,1:-1,2::,2::]-field[2,1:-1,2::,0:-2]-field[2,1:-1,0:-2,2::]+field[2,1:-1,0:-2,0:-2])
        grad_div_y[1:-1,1:-1,1:-1] = self.delta**2*(0.25*yx_A_x + yy_A_y + 0.25*yz_A_z)

        zx_A_x = (field[0,2::,1:-1,2::]-field[0,2::,1:-1,0:-2]-field[0,0:-2,1:-1,2::]+field[0,0:-2,1:-1,0:-2])
        zy_A_y = (field[1,1:-1,2::,2::]-field[1,1:-1,2::,0:-2]-field[1,1:-1,0:-2,2::]+field[1,1:-1,0:-2,0:-2])
        zz_A_z = (field[2,1:-1,1:-1,2::]-2*field[2,1:-1,1:-1,1:-1]+field[2,1:-1,1:-1,0:-2])
        grad_div_z[1:-1,1:-1,1:-1] = self.delta**2*(0.25*zx_A_x + 0.25*zy_A_y + zz_A_z)

        return np.array([grad_div_x,grad_div_y,grad_div_z])

    def curl(self,field):
        curl_x = self.directionalDeriv(field, component=2, direction=1) - self.directionalDeriv(field, component=1,
                                                                                    direction=2)
        curl_y = self.directionalDeriv(field, component=0, direction=2) - self.directionalDeriv(field, component=2,
                                                                                    direction=0)
        curl_z = self.directionalDeriv(field, component=1, direction=0) - self.directionalDeriv(field, component=0,
                                                                                    direction=1)
        curl = np.array(curl_x,curl_y,curl_z)
        return curl

    def directionalDeriv(self,field_component,component=0,direction=0):
        if direction == 0:
            deriv_x = np.zeros((3,self.grid_size,self.grid_size,self.grid_size))
            deriv_x[1:-1,1:-1,1:-1] = (field_component[component,2::,1:-1,1:-1]-field_component[component,0:-2,1:-1,1:-1])/(2*self.delta)
            return deriv_x

        elif direction == 1:
            deriv_y = np.zeros((3, self.grid_size, self.grid_size, self.grid_size))
            deriv_y[1:-1,1:-1,1:-1] = (field_component[component,1:-1,2::,1:-1]-field_component[component,1:-1,0:-2,1:-1])/(2*self.delta)
            return deriv_y

        elif direction == 2:
            deriv_z = np.zeros((3,self.grid_size,self.grid_size,self.grid_size))
            deriv_z[1:-1,1:-1,1:-1] = (field_component[component,1:-1, 1:-1, 2::] - field_component[component, 1:-1, 1:-1, 0:-2]) / (2 * self.delta)
            return deriv_z
        else:
            print("direction is not within 0-2. direction=",direction)
