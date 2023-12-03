"""Main module."""


import numpy as np
import matplotlib.pyplot as plt
# Load the machine
import os
from os.path import join
import scipy.optimize as sopt
import json


class spm_designer:

    def __init__(self, D_s_ext, Length):

        self.D_s_ext = D_s_ext # machine rotor outer diameter (m)
        self.R_s_ext = D_s_ext/2
        self.Lenght_s = Length # stack length  (m)
        self.N_pp = 3      # pair of poles  
        self.N_slots = 36  # number of slots   
        self.N_phases = 3

        self.x = 0.4 # design factor  x = (R_r_int + l_m)/R_s_ext
        self.b = 4.5 # design factor  b = l_m/g

        self.g = 1e-3   # minimum airgap  (m)
        self.k_j = 9.1e3 # thermal loading (W/m2)
        self.N_s = 120 # number of turns per phase

        self.B_fe = 1.5 # Steel loading (T)
        self.B_r = 1.16 # Magnet remanence (T)

        self.beta = 1.0
        self.alpha_m = np.deg2rad(171.0)

        self.Tau_ref = 8.0

        self.rho_Cu = 1.68e-8
        self.k_Cu = 0.432
        self.k_w = 0.95

        self.mu_0 = 4e-7*np.pi

        self.Ntcoil = 20

        self.mu_r = 1.04  # magnet permeability
        self.k_c = 1.15

        self.N_eps = 1000 # number of points to consider when computing B

        self.svg_folder = './svg'
        isExist = os.path.exists(self.svg_folder)
        if not isExist:
            os.makedirs(self.svg_folder)

        self.half = False
        
    def design(self):

        R_s_ext = self.R_s_ext
        g = self.g
        l_m = self.b*self.g
        self.R_r_ext = self.x*R_s_ext - l_m
        self.R_s_int = self.R_r_ext + g + l_m
        self.D_s_int = 2*self.R_s_int
        D_s_ext =  2*self.R_s_ext
        p = self.N_pp
        q = self.N_slots
        B_fe = self.B_fe
        x = self.x
        R_r_ext = self.R_r_ext
        R_s_int = self.R_s_int
        self.l_m = l_m

        self.N_coils = self.N_slots/2

        self.Ntcoil = self.N_s/(self.N_slots/3)*2

        self.rounded_magnet_radius()
        epsilon = np.linspace(-np.pi/2,-np.pi/2+2*np.pi, self.N_eps)
        l_m_eps,g_eps,B_g_eps,B_g_avg,B_g_1,B_g_fft = self.b_eval(epsilon)
        self.B_g_avg = B_g_avg
        self.B_g_1 = B_g_1

        l_y = np.pi*D_s_ext/(4*p*B_fe)*(self.x*B_g_avg) # yoke thickness paper

        D_r_ext = 2*self.R_r_ext
        l_y = B_g_avg/B_fe*np.pi*D_r_ext/(4*p) # yoke thickness ty (Pellegrino:16, pg. 66)

        l_t = R_s_ext - l_y - R_s_int 

        w_t = np.pi*D_s_ext/(6*p*q*B_fe)*(self.x*B_g_avg) # aquí el denominador es muy grande
        w_t = np.pi*D_s_ext/q*(self.x*B_g_avg/B_fe)       # quizás confundieron q (número de slots según paper con q según syre)

        self.l_y = l_y
        self.l_t = l_t
        self.w_t = w_t

        l_end = (self.D_s_int+5*l_t)*np.pi/(p*q)
        l_end = (self.D_s_int+5*l_t)*np.pi/4

        #l_end = 2*(self.R_s_int)*np.pi/6
        self.l_end = l_end
  
        Circumference = np.pi*(D_s_ext + 2*R_s_int)/2
        Slot_width = (Circumference/q - w_t)

        self.Slot_width = Slot_width
        self.w_s = Slot_width
        self.w = self.w_s/(self.w_s + self.w_t) 

        # for slot SlotW11:

        self.SlotW11_Zs = self.N_slots # Slot number
        self.SlotW11_H0 = self.l_t*0.05  # Slot isthmus height
        self.SlotW11_H1 = self.l_t*0.02 # Height
        self.SlotW11_H2 = self.l_t # Slot height below wedge 
        self.SlotW11_W1 = np.pi*(R_s_int+self.SlotW11_H0)/q # Slot top width
        self.SlotW11_W2 = np.pi*(R_s_int+self.SlotW11_H0+l_t)/q # Slot bottom width
        self.SlotW11_W0 = self.SlotW11_W1 * 0.45  # Slot isthmus width
        self.SlotW11_R1 = self.SlotW11_W2*0.25 # Slot bottom radius

        self.A_slot = (self.SlotW11_W1 + self.SlotW11_W2)/2 * l_t
        A_slots = q*self.A_slot 

        k_Cu = self.k_Cu
        rho_Cu = self.rho_Cu
        L = self.Lenght_s
        k_j = self.k_j
        N_s = self.N_s
        I = 1/(6*N_s)*np.sqrt(k_j*(k_Cu/rho_Cu*L/(L+l_end)*2*np.pi*D_s_ext*A_slots))
 
        k_w = self.k_w
        lambda_m = 2*(R_s_int)*L*N_s*k_w*B_g_1/p

        mu_0 = self.mu_0
        k_s = 1.5 # Pyrhonen, Juha, Design of rotating electrical machines, 2014, pg 250
        k_t = 0.787    # Pyrhonen, Juha, Design of rotating electrical machines, 2014, pg 250
        k_c = 1.1
        L_m = 3/2*8/np.pi*(k_w*N_s/p)**2*mu_0*L*D_s_ext*x/g/(l_m/g+k_c)

        Lmd = (6/np.pi*mu_0)/p**2*R_r_ext*L/(self.b*g+g*k_c)*(N_s*k_w)**2  #[mH] Juha Pyrhonen 'Design of rotating electrical machines' (3.110)
        L_m = Lmd

        L_slot = 12/(6*p*4)*k_s*mu_0*L*N_s**2
        L_tip = 12/(6*p*4)*k_t*mu_0*L*N_s**2

        L_q = (L_m + L_slot + L_tip)
        L_d = L_q
        L_s = L_q

        i_q = I
        PF = lambda_m/(np.sqrt(lambda_m**2+(L_q*i_q)**2))

        #L_m = 3/2*8/np.pi*(k_w*N_s/p)**2*mu_0*L*D*x/(l_m+k_c) # esto esta mal
        T_e = 3/2*p*lambda_m*i_q
    
        k_j_computed = (6*N_s*I)**2/(k_Cu/rho_Cu*L/(L+l_end)*2*np.pi*D_s_ext*A_slots)

        self.K_s = N_s*k_w*l_t*8*k_Cu*self.w_s/(self.w_s+w_t)

        self.T_e = T_e
        self.PF = PF

        self.I = I
        self.k_j_computed = k_j_computed
        self.lambda_m = lambda_m
        self.L_m = L_m
        self.L_d = L_d
        self.L_q = L_q
        self.L_s = L_s
        self.L_slot = L_slot
        self.L_tip = L_tip

        self.Vol_r = np.pi*self.R_r_ext**2 * self.Lenght_s

        self.I_density = self.Ntcoil*I*np.sqrt(2)/self.A_slot

        self.wire_eval()

    def wire_eval(self):

        Ntcoil = self.Ntcoil
        self.wire_area = self.A_slot*self.k_Cu/Ntcoil
        self.wire_radio = np.sqrt(self.wire_area/np.pi) # A = pi*r**2
        self.wire_diameter = 2*self.wire_radio

        N_slots = self.N_slots
        Lenght_s = self.Lenght_s
        
        Nlay = 1
        #Lew = 2*(self.R_s_int)*np.pi/6
        Lew = self.l_end
        self.phase_wire_length = (0.5 * 2 * Lew + Lenght_s) * N_slots * Nlay * Ntcoil/3

        self.Ntcoil = Ntcoil
        self.Nlay = 1
        self.Lew = Lew
 
        # estimated phase resistance
        self.R_s = self.rho_Cu*self.phase_wire_length/self.wire_area


    def rounded_magnet_radius(self):
        '''
        Compute the radius of the rounded magnet.

        '''

        R_r_ext = self.R_r_ext
        beta = self.beta
        l_m = self.l_m
        alpha_m = self.alpha_m
        alpha_m_2 = alpha_m/self.N_pp
        
        cos,sin = np.cos,np.sin
        num = (2*R_r_ext**2 + 2*l_m*R_r_ext*(beta+1))*(1 - cos(alpha_m_2/2)) + (beta**2 + 1 - 2*beta*cos(alpha_m_2/2))*l_m**2
        den = 2*(R_r_ext*(1-cos(alpha_m_2/2))+l_m*(1-beta*cos(alpha_m_2/2)))
        r_c = num/den
        self.r_c = r_c

    def b_eval(self, epsilon):
        '''
        Compute:
         
          - the B curve as function of angle
          - B average
          - B fundamental fourier value

        '''
 

        B_r = self.B_r
        mu_r = self.mu_r 
        k_c = self.k_c 
        g = self.g
        N_eps = len(epsilon)
        
        R_r_ext = self.R_r_ext
        R_s_int = self.R_s_int 
        beta = self.beta
        l_m = self.l_m
        alpha_m_r = self.alpha_m/self.N_pp # actual geometrical angle
        alpha_m_e = self.alpha_m # electrical angle

        alpha_m_2 = alpha_m_e
        
        cos,sin = np.cos,np.sin
        num = (2*R_r_ext**2 + 2*l_m*R_r_ext*(beta+1))*(1 - cos(alpha_m_2/2)) + (beta**2 + 1 - 2*beta*cos(alpha_m_2/2))*l_m**2
        den = 2*(R_r_ext*(1-cos(alpha_m_2/2))+l_m*(1-beta*cos(alpha_m_2/2)))
        r_c = num/den

        cos, sin = np.cos,np.sin
        
        idx_pos_left  = np.abs(epsilon-(-alpha_m_e/2)).argmin()
        idx_pos_right = np.abs(epsilon-(alpha_m_e/2)).argmin()
        idx_half = np.abs(epsilon-np.pi/2).argmin()
        idx_neg_left  = np.abs(epsilon-(-alpha_m_e/2+np.pi)).argmin()
        idx_neg_right = np.abs(epsilon-( alpha_m_e/2+np.pi)).argmin()

        # Magnet thickness as function of epsilon
        epsilon_half = epsilon[:idx_half]
        l_m_eps = epsilon*0.0
        l_m_eps_half = (R_r_ext+l_m-r_c)*cos(epsilon_half) - R_r_ext + np.sqrt(r_c**2 - ((R_r_ext+l_m)*sin(epsilon_half)-r_c*sin(epsilon_half))**2)
        l_m_eps_half[0:idx_pos_left] = 0.0
        l_m_eps_half[idx_pos_right:] = 0.0

        if not self.half:
            l_m_eps[:idx_half] = l_m_eps_half
            l_m_eps[idx_half:] = l_m_eps_half
        else:
            l_m_eps = l_m_eps_half

        # Airgap as function of epsilon
        g_eps = R_s_int - R_r_ext - l_m_eps

        # B as function of epsilon
        B_g_eps = l_m_eps/g_eps/(l_m_eps/g_eps+k_c*mu_r)*B_r

        B_g_eps[0:np.argmin(epsilon<-alpha_m_e/2)] = 0
        B_g_eps[np.argmin(epsilon<alpha_m_e/2):] = 0
        if not self.half:
            B_g_eps[int(N_eps/2):] = -B_g_eps[:int(N_eps/2)]

        # B average
        B_g_avg = np.sum(np.abs(B_g_eps))/N_eps

        # B fourier components
        B_g_fft = np.abs(np.fft.rfft(B_g_eps))/N_eps*2
        B_g_1 = B_g_fft[1]

        return l_m_eps,g_eps,B_g_eps,B_g_avg,B_g_1,B_g_fft
    
    def plot_b(self):
        self.half = False
        epsilon = np.linspace(-np.pi/2,-np.pi/2+2*np.pi, self.N_eps)
        l_m_eps,g_eps,B_g_eps,B_g_avg,B_g_1,B_g_fft = self.b_eval(epsilon)
        fig, axes = plt.subplots(ncols=2, figsize=(10,4))
        axes[0].plot(np.rad2deg(epsilon),B_g_eps)
        axes[0].plot([-90,90],[B_g_avg]*2)
        axes[0].set_ylabel('Airgap induction (T)')
        axes[0].set_xlabel('Electrical angle (º)')
        axes[0].set_xlim([-90,270])
        axes[0].grid()
        axes[1].stem(B_g_fft[0:10])
        axes[1].grid()
        axes[1].set_ylabel('Airgap induction (T)')
        axes[1].set_xlabel('Fourier coefficient')
        fig.savefig(os.path.join(self.svg_folder,'b_harmonics.svg'))
        return fig 
        
        
    def plot_machine(self):
        
        # x/R_r = sin(epsilon)
        # y/R_r = sin(epsilon)

        self.half = True

        epsilon = np.linspace(-np.pi/2,2*np.pi-np.pi/2, 200)
        #epsilon_beta = np.linspace(-self.alpha_m/2,self.alpha_m/2, 200)/self.N_pp
        epsilon_beta = epsilon
        R_s_int = self.R_s_int
        x_s_int = R_s_int*np.sin(epsilon)
        y_s_int = R_s_int*np.cos(epsilon)

        R_s_l_t_int = self.R_s_int + self.l_t
        x_s_l_t_int = R_s_l_t_int*np.sin(epsilon)
        y_s_l_t_int = R_s_l_t_int*np.cos(epsilon)

        R_s_ext = self.R_s_ext
        x_s_ext = R_s_ext*np.sin(epsilon)
        y_s_ext = R_s_ext*np.cos(epsilon)
        x_s_ext_1 = R_s_ext*np.sin(epsilon[0])
        y_s_ext_1 = R_s_ext*np.cos(epsilon[0])
        x_s_ext_2 = R_s_ext*np.sin(epsilon[-1])
        y_s_ext_2 = R_s_ext*np.cos(epsilon[-1])

        R_r_ext = self.R_r_ext
        x_r_ext = R_r_ext*np.sin(epsilon)
        y_r_ext = R_r_ext*np.cos(epsilon)

        l_m_eps,g_eps,B_g_eps,B_g_avg,B_g_1,B_g_fft = self.b_eval(epsilon_beta)
        x_beta = (R_r_ext+l_m_eps)*np.sin(epsilon_beta)
        y_beta = (R_r_ext+l_m_eps)*np.cos(epsilon_beta)
        x_beta_11 = (R_r_ext)*np.sin(epsilon_beta[0])
        y_beta_11 = (R_r_ext)*np.cos(epsilon_beta[0])
        x_beta_12 = (R_r_ext+l_m_eps[0])*np.sin(epsilon_beta[0])
        y_beta_12 = (R_r_ext+l_m_eps[0])*np.cos(epsilon_beta[0])
        x_beta_21 = (R_r_ext)*np.sin(epsilon_beta[-1])
        y_beta_21 = (R_r_ext)*np.cos(epsilon_beta[-1])
        x_beta_22 = (R_r_ext+l_m_eps[-1])*np.sin(epsilon_beta[-1])
        y_beta_22 = (R_r_ext+l_m_eps[-1])*np.cos(epsilon_beta[-1])

        fig, axes = plt.subplots(ncols=1, figsize=(10,4))
        axes.plot(x_r_ext,y_r_ext,'k')
        axes.plot(x_beta,y_beta,'k')
        axes.plot(x_s_int,y_s_int)
        axes.plot(x_s_ext,y_s_ext)
        axes.plot(x_s_l_t_int,y_s_l_t_int)
        axes.plot([0,x_s_ext_1],[0,y_s_ext_1],'k', lw = 0.1)
        axes.plot([0,x_s_ext_2],[0,y_s_ext_2],'k', lw = 0.1)
        axes.plot([0,x_s_ext_1],[0,y_s_ext_1],'k', lw = 0.1)
        axes.plot([x_beta_11,x_beta_12],[y_beta_11,y_beta_12],'k')
        axes.plot([x_beta_21,x_beta_22],[y_beta_21,y_beta_22],'k')

        return fig
        

    def plot_xb_plane(self,x_min=0.5,x_max=0.7,b_min=3.5,b_max=5.5, N_x=50, N_b=50):

        x = np.linspace(x_min,x_max, N_x)
        b = np.linspace(b_min,b_max, N_b)
        T_e = np.zeros((N_x,N_b))
        PF  = np.zeros((N_x,N_b))

        X, B = np.meshgrid(x, b)

        for itx in range(N_x):
            for itb in range(N_b):
                self.x = X[itx,itb]
                self.b = B[itx,itb]
                self.design()
                self.wire_eval()
                T_e[itx,itb] = self.T_e
                PF[itx,itb] = self.PF
        
        self.X = X
        self.B = B
        self.T_e_array = T_e
        self.PF_array = PF

        fig, ax = plt.subplots(figsize=(6,4));
        fig.get_tight_layout()

        T_e_contour = ax.contour(self.X, self.B, self.T_e_array, cmap='cool');
        PF_contour  = ax.contour(self.X, self.B, self.PF_array, cmap='summer');
        pa = ax.clabel(T_e_contour, inline=True, fontsize=8)
        pb = ax.clabel(PF_contour, inline=True, fontsize=8)
        # cba = plt.colorbar(pa,shrink=0.25)
        # cbb = plt.colorbar(pb,shrink=0.25)
        fig.colorbar(T_e_contour, label='Torque (Nm)')
        fig.colorbar(PF_contour, label='Power factor (-)')

        #ax.set_title('x-b plane')
        ax.set_ylabel('$l_m/g$')
        ax.set_xlabel('$x = (R_r^{int} + l_m)/R_s^{ext}$') 
        fig.savefig(os.path.join(self.svg_folder,'plane_x_b.svg'))


    def report_pellegrino(self):

        print(f'x = {self.x:0.3f}, b = {self.b:0.3f}')
        print(f'L_m = {self.L_m*1e3}, L_slot = {self.L_slot*1e3}, L_tip = {self.L_tip*1e3}')
        print(f'I = {self.I:0.2f} A, PF = {self.PF:0.3f}, T_e = {self.T_e:0.3f} Nm')
        print(f'R_r_ext = {self.R_r_ext*1e3:0.2f} mm, R_s_ext = {self.R_s_ext*1e3:0.2f} mm')
        print(f'yoke l_y = {self.l_y*1e3:0.2f} mm, slot l_t = {self.l_t*1e3:0.2f} mm, slot w_t = {self.w_t*1e3:0.2f} mm')
        print(f'magnet l_m = {self.l_m*1e3:0.2f} mm')
        print(f'L_q = {self.L_q*1000:0.2f} mH, R_s = {self.R_s:0.3f} Ohm,  A_slot = {self.A_slot*1e6:0.2f} mm2')
        #print(f'l_end = {self.l_end*1000} mm')

        # print(f'PF = {self.PF:0.3f}')
        print(f'r_c = {self.r_c}')
        # print(f'k_j_computed = {self.k_j_computed:0.3f} J, K_s = {self.K_s:0.3f} A/m')
        print(f'B_g_avg = {self.B_g_avg:0.3f} T, B_g_1 = {self.B_g_1:0.3f} T, lambda_m = {self.lambda_m:0.3f} Vs')
        # print(f'w_t = {self.w_t*1000:0.3f} mm, w_s = {self.w_s*1000:0.3f} mm, w = {self.w:0.3f}')
        # 

    def inductance_eval(self):

        mu_0 = np.pi*4e-7
        R_r_ext = self.R_r_ext
        R_s_int = self.R_s_int

        N_pp = self.N_pp
        Length = self.Lenght_s
        A = 2*R_r_ext*np.pi/(2*N_pp)*Length
        N = self.N_s/N_pp

        # L = phi/I 
        # phi = N*I/Rel
        # Rel = l/(mu*A)
        l = 2*(R_s_int-R_r_ext)
        L = N_pp*mu_0*A*N**2/l
        print(f'Computed inductance = {L}')

    def save2pyleecan(self,file):

        data = {'p':self.N_pp,'stator':{'lamination':{},'slot':{},'winding':{},'conductor':{}},
                'rotor':{'lamination':{},'slot':{}, 'magnet':{}},
                'results':{}}

        data['stator']['lamination'].update({
            'Rint':self.R_s_int, # internal radius [m]
            'Rext':self.R_s_ext, # external radius [m]
            'L1':self.Lenght_s, # Lamination stack active length [m] without radial ventilation airducts 
                        # but including insulation layers between lamination sheets
            'Nrvd':0, # Number of radial air ventilation duct
            'Kf1':self.k_w, # Lamination stacking / packing factor
            'is_internal':False,
            'is_stator':True,
            })

        data['stator']['slot'].update({'type':'SlotW11'})
        data['stator']['slot'].update({
        'Zs': self.SlotW11_Zs , # Slot number
        'H0': self.SlotW11_H0 , # Slot isthmus height
        'H1': self.SlotW11_H1 , # Height
        'H2': self.SlotW11_H2 , # Slot height below wedge 
        'W0': self.SlotW11_W0 , # Slot isthmus width
        'W1': self.SlotW11_W1 , # Slot top width
        'W2': self.SlotW11_W2 , # Slot bottom width
        'R1': self.SlotW11_R1   # Slot bottom radius
        })

        data['stator']['winding'].update({
            'qs':3,  # number of phases
            'p':self.N_pp,  # number of pole pairs
            'Nlayer':1,  # Number of layers per slots
            'coil_pitch':0, # Coil pitch (or coil span or Throw)
            'Lewout':self.Lew,  # staight length of conductor outside lamination before EW-bend
            'Ntcoil':self.Ntcoil,  # number of turns per coil
            'Npcp':1,  # number of parallel circuits per phase
            'Nslot_shift_wind':0,  # 0 not to change the stator winding connection matrix built by pyleecan number 
                                # of slots to shift the coils obtained with pyleecan winding algorithm 
                                # (a, b, c becomes b, c, a with Nslot_shift_wind1=1)
            'is_reverse_wind':False # True to reverse the default winding algorithm along the airgap 
                                # (c, b, a instead of a, b, c along the trigonometric direction)
        })

        data['stator']['conductor'].update({
                    'type':'CondType12',
            'Wwire':self.wire_diameter, #  single wire width without insulation [m]
            # Hwire=2e-3, # single wire height without insulation [m]
            'Wins_wire':1e-6, # winding strand insulation thickness [m]
            #type_winding_shape=1, # type of winding shape for end winding length calculation
            #                       # 0 for hairpin windings
            #                       # 1 for normal windings
                })        
 

        # Rotor setup
        data['rotor']['lamination'].update({
            'Rint': 0.0, # Internal radius
            'Rext': self.R_r_ext, # external radius
            'is_internal':True, 
            'is_stator':False,
            'L1':self.Lenght_s # Lamination stack active length [m] 
                        # without radial ventilation airducts but including insulation layers between lamination sheets
        })

        data['rotor']['slot'].update({
            'type':'SlotM14',
            'H0' : 0.0001,
            'Hmag': self.l_m,
            'Rtopm': self.r_c,
            'W0':np.pi/(self.N_pp),
            'Wmag':self.alpha_m/(self.N_pp),
            'Zs':self.N_pp*2
                   })
        data['I'] = self.I

        data['rotor']['magnet'].update({'mur_lin':1.0})
        data['rotor']['magnet'].update({'Brm20':self.B_r})

        data['results'].update({'L_q':self.L_q})
        data['results'].update({'R_s':self.R_s})
        data['results'].update({'B_g_avg':self.B_g_avg})
        data['results'].update({'B_g_1':self.B_g_1})
        data['results'].update({'lambda_m':self.lambda_m})
        data['results'].update({'T_e':self.T_e})
        data['results'].update({'K_s':self.K_s})
        data['results'].update({'k_j_computed':self.k_j_computed})
        data['results'].update({'I_density':self.I_density})

        self.data = data

        with open(file,'w') as fobj:
            fobj.write(json.dumps(self.data))

    def torquepf2xb(self, T_e_ref, PF_ref):
        self.T_e_ref = T_e_ref
        self.PF_ref = PF_ref

        xb_0 = [0.5, 4.5]
        result = sopt.minimize(self.obj_xb, xb_0, bounds=[(0.3,0.7),(3.5,5.5)])

        self.opt_results = result

        self.x = result.x[0]
        self.b = result.x[1]
        print(f'x = {self.x:0.2f}, b = {self.b:0.2f}')

    def obj_xb(self,xb):

        x = xb[0]
        b = xb[1]

        self.x =x # design factor  x = (R_r_int + l_m)/R_s_ext
        self.b =b # design factor  b = l_m/g

        self.design()
        self.wire_eval()

        return 1000*(self.T_e-self.T_e_ref)**2 + 1000*( self.PF - self.PF_ref)**2

               


if __name__ == '__main__':

    spm = spm_designer(0.175,0.11)
    spm.beta = 0.5
    spm.g = 1e-3
    spm.k_j = 9100.0
    spm.x = 0.6 # design factor  x = (R_r_int + l_m)/R_s_ext
    spm.b = 4.5 # design factor  b = l_m/g

    spm.design()
    
    spm.report_pellegrino()
    #spm.plot_machine()
    spm.plot_xb_plane()
    spm.plot_b()
    #spm.save2pyleecan('dicorca_spm.json')
    spm.torquepf2xb(57,0.93)
    spm.design()
    spm.wire_eval()
    spm.report_pellegrino()

