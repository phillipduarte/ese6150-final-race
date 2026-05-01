import numpy as np
import matplotlib.pyplot as plt

# Load raceline
data = np.loadtxt('raceline.csv', delimiter=';', comments='#')
s        = data[:, 0]
x        = data[:, 1]
y        = data[:, 2]
psi      = data[:, 3]
speed    = data[:, 5]

# Precompute normals (do this once at startup)
nx = -np.sin(psi)
ny =  np.cos(psi)
total_length = s[-1]

i = len(x) // 2

# Construct test points on both sides
left_x  = x[i] + 0.5 * nx[i]
left_y  = y[i] + 0.5 * ny[i]
right_x = x[i] - 0.5 * nx[i]
right_y = y[i] - 0.5 * ny[i]

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'b-', label='raceline')
plt.plot(x[i], y[i], 'ko', markersize=8, label='test waypoint')
plt.plot(left_x,  left_y,  'g^', markersize=10, label='+nx (should be left)')
plt.plot(right_x, right_y, 'rv', markersize=10, label='-nx (should be right)')

# Draw an arrow showing direction of travel at that point
plt.annotate('', xy=(x[i] + 0.3*np.cos(psi[i]), y[i] + 0.3*np.sin(psi[i])),
             xytext=(x[i], y[i]),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))

plt.axis('equal')
plt.legend()
plt.title('Normal vector sanity check')
plt.show()

def cart_to_frenet(px, py):
    # Find closest waypoint
    dists = np.sqrt((x - px)**2 + (y - py)**2)
    i = np.argmin(dists)
    
    # s: use precomputed arc length
    s_out = s[i]
    
    # d: signed lateral offset via normal vector
    dx = px - x[i]
    dy = py - y[i]
    d_out = dx * nx[i] + dy * ny[i]
    
    return s_out, d_out

def frenet_to_cart(s_in, d_in):
    # Wrap s for closed loop
    s_wrapped = s_in % total_length
    
    # Binary search for waypoint index
    i = np.searchsorted(s, s_wrapped, side='right') - 1
    i = np.clip(i, 0, len(s) - 1)
    
    # Reconstruct Cartesian position
    px = x[i] + d_in * nx[i]
    py = y[i] + d_in * ny[i]
    
    return px, py