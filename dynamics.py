def next_state(part, Walls, state_history):
    # get dts from each wall; each wall may return 0, 1, or many.  M = most dt's reported by a single wall
    L = [len(wall.get_dt(part)) for wall in Walls]  # computes dts at each wall and records how many dts each wall reports
    N = max(L)
    M = len(Walls)
    DT = np.full(shape=(M,N), fill_value=np.inf)  # array to hold dts from wall i in row i
    for i, l in enumerate(L):
        DT[i, :l] = Walls[i].dt  # write dt's from wall i in row i

    DT[DT < 0] = np.inf  # set any negative's to np.inf

    # attempt to move using the smallest dt and check if its a real collision.  If not, move back, set that dt to np.inf, and try again
    for attempt in range(100):
        row, col = np.unravel_index(DT.argmin(), DT.shape)  # What slot contains the smallest positive time
        part.dt = DT[row,col]
        part.wall_idx = row
        next_wall = Walls[row]

        # Move particle
        part.pos += part.vel * part.dt
        part.t += part.dt

        # check if this is a real or false collision (false = against part of wall that has been removed)
        if part.check_real_collision_get_arclength():  # if real collision, great!  We found next_state
            next_wall.resolve_collision(part, state_history)
            part.get_phi()
            break
        else:  # if not real collision, move the particle back and try again with next smallest dt
            part.pos -= part.vel * part.dt
            part.t -= part.dt
            DT[row, col] = np.inf

    return part, Walls


def record_state(part, history=None):
    if history is None:  # create history
        history = {'POS':[],
                   'VEL':[],
                   'SPIN':[],
                   'WALL':[],
                   'T':[],
                   'WRAP_COUNT':[],
                   'PHI':[],
                   'ARCLENGTH':[],
                  }

    history['POS'].append(       part.pos.copy())
    history['VEL'].append(       part.vel.copy())
    history['SPIN'].append(      part.spin)
    history['WALL'].append(      part.wall_idx)
    history['T'].append(         part.t)
    history['WRAP_COUNT'].append(part.wrap_count)
    history['PHI'].append(       part.phi)
    history['ARCLENGTH'].append( part.arclength)
    return history

def draw(history, Walls, start_step=0, stop_step=None, ax=None, arrows=True):
    pos_hist = history['POS'][start_step:stop_step]
    pos = pos_hist[-1]

    vel_hist = history['VEL'][start_step:stop_step]
    vel = vel_hist[-1]

    t_hist = history['T'][start_step:stop_step]
    t = t_hist[-1]

    try:
        rot_hist = history['ROT'][start_step:stop_step]
        rot = rot_hist[-1]
    except:
        pass

    if ax is None:   # May pass in ax to overlay plots
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_aspect('equal')
        ax.grid(False)

    for wall in Walls:
        ax.plot(*wall.get_bdy_pts(), 'k', linewidth=3.0)

    bdy_pts = part.get_bdy_pts()
    try:
        bdy_pts = rot @ bdy_pts
    except:
        pass
    bdy_pts = bdy_pts + pos[:,np.newaxis]
    ax.plot(*bdy_pts)

#     Draw arrow for velocity
    arr = vel / np.linalg.norm(vel) * part.radius * 2
    ax.annotate("", xy=pos, xytext=pos+arr, arrowprops=dict(arrowstyle="<-"))

    steps = len(pos_hist)
    if steps > 1:
        dt = np.abs(np.diff(t_hist))
        big_steps = np.nonzero(dt > tol)[0].tolist()
        for i in big_steps:
            ax.plot(pos_hist[i:i+2,0], pos_hist[i:i+2,1], 'g:')
        if arrows:
            midpoints = (pos_hist[1:] + pos_hist[:-1]) / 2
            vec = pos_hist[1:] - pos_hist[:-1]
            mag = np.linalg.norm(vec, axis=1, keepdims=True)
            vec /= mag
            vec *= (mag.min() / 2)
            for i in big_steps:
                ax.quiver(midpoints[i,0], midpoints[i,1], vec[i,0], vec[i,1])
    return ax