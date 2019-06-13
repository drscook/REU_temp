def next_state(part, Walls):
    # get dts from each wall and figure out which wall reported the most 
    M = max([len(wall.get_dt(part)) for wall in Walls])
    
    # pad all wall.dt to length M using np.inf so we can stack them into a single array
    DT = np.array([np.concatenate((wall.dt, np.full(shape=M-len(wall.dt), fill_value=np.inf))) for wall in Walls])
    
    # set any negative's to np.inf
    DT[DT < 0] = np.inf

    # attempt to move using the smallest dt and check if its a real collision.  If not, move back, set that dt to np.inf, and try again
    for attempt in range(100):
        row, col = np.unravel_index(DT.argmin(), DT.shape)  # What slot contains the smallest positive time
        part.dt = DT[row,col]
        part.wall_idx = row
        next_wall = Walls[row]

        # Move particle
        part.pos += part.vel * part.dt
        part.t += part.dt

        if part.check_real_collision_get_arclength():  # if real collision, great!  We found next_state
            next_wall.resolve_collision(part)
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
                   'WALL':[],
                   'T':[],
                   'WRAP_COUNT':[],
                   'PHI':[],
                   'ARCLENGTH':[],
                  }

    history['POS'].append(       part.pos.copy())
    history['VEL'].append(       part.vel.copy())
    history['WALL'].append(      part.wall_idx)
    history['T'].append(         part.t)
    history['WRAP_COUNT'].append(part.wrap_count)
    history['PHI'].append(       part.phi)
    history['ARCLENGTH'].append( part.arclength)
    return history



def draw(history, Walls, start_step=0, stop_step=None, ax=None):
    pos_hist = np.array(history['POS'][start_step:stop_step])
    vel_hist = np.array(history['VEL'][start_step:stop_step])
    
    pos = pos_hist[-1]
    vel = vel_hist[-1]
    
    if ax is None:   # May pass in ax to overlay plots
        fig, ax = plt.subplots(figsize=(5,5))

    for wall in Walls:
        pts = wall.draw_me()
        ax.plot(*pts, 'k', linewidth=3.0)

    pts = part.draw_me() + pos[:,np.newaxis]
    ax.plot(*pts)
        
#     Draw arrow for velocity
    ax.annotate("", xy=pos, xytext=pos+vel, arrowprops=dict(arrowstyle="<-"))
    
    # Draw trails
    if len(pos_hist) > 1:
        ax.plot(pos_hist[:,0], pos_hist[:,1], 'g:')
        midpoints = (pos_hist[1:] + pos_hist[:-1]) / 2
        vec = pos_hist[1:] - pos_hist[:-1]
        mag = np.linalg.norm(vec, axis=1, keepdims=True)
        vec /= mag
        vec *= (mag.min() / 2)
        ax.quiver(midpoints[:,0], midpoints[:,1], vec[:,0], vec[:,1])

    ax.set_aspect('equal')
    return ax