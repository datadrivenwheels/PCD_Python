def find_min_rss_like_tau(X, y, y_pred, alpha = 0.05):
    min_like = np.inf  # Changed to positive infinity
    tau = None
    n = len(X)
    for i in range(1, len(X) - 1):  # Exclude the very first and last points as potential taus
        # Calculate RSS for both segments
        RSS_left = i * np.log((1/i) * np.sum((y[:i] - y_pred[:i]) ** 2))
        RSS_right = (len(X) - i) * np.log((1/(len(X) - i)) * np.sum((y[i:] - y_pred[i:]) ** 2))
        
        # Calculate the difference in RSS
        likelihood = RSS_left + RSS_right  # Removed np.abs as we are looking for the minimum
        
        # Update tau and min_diff if this tau gives a smaller difference
        if likelihood < min_like:  # Changed to less than for minimization
            min_like = likelihood
            tau = X[i]

    # test
    likelihood_n = len(X) * np.log((1/len(X)) * np.sum((y - y_pred) ** 2))
    delta_n = np.sqrt(likelihood_n - min_like)

    a_n = np.sqrt(2 * np.log(np.log(n))) / np.log(n)
    b_n = (2 * np.log(np.log(n)) + 0.5 * np.log(np.log(np.log(n))) - np.log(np.pi)) / np.log(n)

    if a_n * delta_n * np.sqrt(np.log(n)) - b_n * np.log(n) > - np.log(-np.log((1 - alpha) / 2)):
        return tau, min_like
    else:
        return None, None

    #return tau_final, min_like
    
def changing_point_p_spline(x, y, alpha = 0.05):
    # Convert x to a 2D array for compatibility with pyGAM
    X = x.reshape(-1, 1)
    # Fitting a penalized spline model with pyGAM
    gam = LinearGAM(s(0, n_splines=10, spline_order=3, penalties='auto')).fit(X, y)
    # Generating points to plot the fitted model
    y_pred = gam.predict(X)
    tau, value = find_min_rss_like_tau(X, y, y_pred, alpha)
    return tau, y_pred, value

def recursive_change_point_detection(x, y, alpha=0.05, change_points=None, min_segment_size=130):
    if change_points is None:
        change_points = []
    
    # Base case: if segment is too small, return
    if len(x) < min_segment_size:
        return change_points
    
    # Find change point in current segment
    tau, y_pred, value = changing_point_p_spline(x, y, alpha)
    
    # If no significant change point found, return current list of change points
    if tau is None:
        return sorted(change_points)
    
    # Add the found change point (adjust by segment start position)
    tau_idx = np.where(x == tau[0])[0][0]
    change_points.append(tau[0])
    
    # Recursively find change points in left and right segments
    left_points = recursive_change_point_detection(x[:tau_idx], y[:tau_idx], 
                                                 alpha, change_points, min_segment_size)
    right_points = recursive_change_point_detection(x[tau_idx:], y[tau_idx:], 
                                                  alpha, change_points, min_segment_size)
    
    # Combine and sort all change points
    all_points = sorted(set(change_points))
    return all_points

def plot_multiple_change_points(x, y, title, alpha=0.05, show_spline=True):
    # Get all change points
    change_points = recursive_change_point_detection(x, y, alpha)
    
    # Fit spline to entire dataset
    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=10, spline_order=3, penalties='auto')).fit(X, y)
    y_pred = gam.predict(X)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Detection confidence score * IoU',  
               color='orange', marker='o', alpha=0.5, s=50)
    
    if show_spline:
        plt.plot(x, y_pred, label='Penalized Spline Fit', 
                color='darkblue', linestyle='--', linewidth=2)
    
    # Plot all change points with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(change_points)))
    for i, (tau, color) in enumerate(zip(change_points, colors)):
        plt.axvline(x=tau, color=color, linestyle='-.', 
                   linewidth=2, label=f'Var. change point {i+1}')
        print(f"Var. change point {i+1} = {tau}")
    
    plt.xlabel('Distance (m)', fontsize=14)
    plt.ylabel('Detection confidence score * IoU', fontsize=14)
    plt.title(title, fontsize=14, pad=16)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.show()



def work_distance(x, y, change_points, title, weather, prob_threshold=0.5, y_thresh=0.5, show_plot=True):
    """
    Calculate work distance considering multiple change points
    
    Parameters:
    -----------
    x : array-like
        Distance values
    y : array-like
        Detection confidence * IoU values
    change_points : list
        List of change points (can be empty, or contain multiple points)
    title : str
        Plot title
    weather : str
        Weather condition for the title
    prob_threshold : float
        Probability threshold for work distance calculation
    y_thresh : float
        Threshold for detection confidence * IoU
    show_plot : bool
        Whether to show the visualization
    """
    x = np.array(x)
    y = np.array(y)
    
    # Fit spline to the data
    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=10, spline_order=3, penalties='auto')).fit(X, y)
    y_spline = gam.predict(X)
    
    # Create full-sized probability array
    probs = np.ones(len(x))
    
    if not change_points:  # No change points
        sigma = np.std(y)
        if sigma > 0:
            probs = 1 - norm.cdf(y_thresh, loc=y_spline, scale=sigma)
    else:
        # Sort change points to ensure proper segmentation
        change_points = sorted(change_points)
        
        # Create masks for each segment
        segments = []
        
        # First segment (before first change point)
        segments.append(x <= change_points[0])
        
        # Middle segments
        for i in range(len(change_points)-1):
            segments.append((x >= change_points[i]) & (x <= change_points[i+1]))
            
        # Last segment (after last change point)
        segments.append(x >= change_points[-1])
        
        # Calculate probabilities for each segment
        for mask in segments:
            if np.any(mask):  # Check if segment is not empty
                sigma = np.std(y[mask])
                if sigma > 0:
                    probs[mask] = 1 - norm.cdf(y_thresh, loc=y_spline[mask], 
                                             scale=sigma)
                else:
                    probs[mask] = 1 - norm.cdf(y_thresh, loc=y_spline[mask], 
                                             scale=1e-10)  # Small non-zero value
    
    # Calculate work distance
    valid_x = x[probs <= prob_threshold]
    max_x = np.min(valid_x) if valid_x.size > 0 else None
    
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # Plot scatter points
        plt.scatter(x, y, label='IoU*Confidence score',  
                   color='orange', marker='o', alpha=0.5, s=50)
        
        # Plot spline fit
        plt.plot(x, y_spline, label='Penalized Spline Fit', 
                color='darkblue', linestyle='--', linewidth=2)
        
        # Plot threshold line
        plt.axhline(y=prob_threshold, color='sienna', linestyle='-.', linewidth=2, 
                   label=f'y_t = {prob_threshold}')
        
        # Plot change points with different colors
        #colors = plt.cm.rainbow(np.linspace(0, 1, len(change_points)))
        #for i, (tau, color) in enumerate(zip(change_points, colors)):
         #   plt.axvline(x=tau, color=color, linestyle=':', 
          #             linewidth=2, label=f'Change point {i+1}')
        
        # Plot work distance
        if max_x is not None:
            plt.axvline(x=max_x, color='red', linestyle=':', 
                       linewidth=2, label='PCD')
        
        plt.title(f"{title}: {weather}", fontsize=14, pad=15)
        plt.xlabel("Distance (m)", fontsize=15)
        plt.ylabel("IoU*Confidence score", fontsize=15)
        plt.legend(fontsize=13)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        plt.show()
        
    return max_x


def compute_iou(ground_truth, predicted):
    # Get the coordinates of the intersection rectangle
    x1 = max(ground_truth.iloc[0], predicted.iloc[0])
    y1 = max(ground_truth.iloc[1], predicted.iloc[1])
    x2 = min(ground_truth.iloc[2], predicted.iloc[2])
    y2 = min(ground_truth.iloc[3], predicted.iloc[3])
    
    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both the prediction and ground truth rectangles
    ground_truth_area = (ground_truth.iloc[2] - ground_truth.iloc[0] + 1) * (ground_truth.iloc[3] - ground_truth.iloc[1] + 1)
    predicted_area = (predicted.iloc[2] - predicted.iloc[0] + 1) * (predicted.iloc[3] - predicted.iloc[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(ground_truth_area + predicted_area - intersection_area)

    return iou


