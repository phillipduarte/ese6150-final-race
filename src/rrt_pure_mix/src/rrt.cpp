// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well

#include "rrt/rrt.h"

// ============================================================
//  Destructor
// ============================================================
RRT::~RRT() {
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// ============================================================
//  Constructor
// ============================================================
RRT::RRT() : rclcpp::Node("rrt_node"), gen((std::random_device())()) {

    // --- Declare and get ROS parameters ---
    this->declare_parameter("use_rrt_star",        false);
    this->declare_parameter("max_expansion_dist",  0.5);
    this->declare_parameter("goal_threshold",       0.3);
    this->declare_parameter("max_iterations",       500);
    this->declare_parameter("lookahead_dist",       2.0);
    this->declare_parameter("search_radius",        1.0);
    this->declare_parameter("max_shortcut_dist",    1.0);
    this->declare_parameter("goal_bias_prob",       0.10);
    this->declare_parameter("max_speed",            1.5);
    this->declare_parameter("pursuit_lookahead",    0.8);
    this->declare_parameter("inflation_radius",     3);
    this->declare_parameter("waypoints_file",       std::string(""));
    this->declare_parameter("waypoint_speed_scale",  0.8);

    use_rrt_star_       = this->get_parameter("use_rrt_star").as_bool();
    max_expansion_dist_ = this->get_parameter("max_expansion_dist").as_double();
    goal_threshold_     = this->get_parameter("goal_threshold").as_double();
    max_iterations_     = this->get_parameter("max_iterations").as_int();
    lookahead_dist_     = this->get_parameter("lookahead_dist").as_double();
    search_radius_      = this->get_parameter("search_radius").as_double();
    max_shortcut_dist_  = this->get_parameter("max_shortcut_dist").as_double();
    goal_bias_prob_     = this->get_parameter("goal_bias_prob").as_double();
    max_speed_          = this->get_parameter("max_speed").as_double();
    pursuit_lookahead_  = this->get_parameter("pursuit_lookahead").as_double();
    inflation_radius_   = this->get_parameter("inflation_radius").as_int();
    waypoint_speed_scale_ = this->get_parameter("waypoint_speed_scale").as_double();

    // --- Publishers ---
    drive_pub_     = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 1);
    tree_pub_      = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_tree", 1);
    path_pub_      = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_path", 1);
    waypoints_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_waypoints", 1);
    goal_pub_      = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_goal", 1);
    grid_pub_      = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rrt_occupancy_grid", 1);

    // --- Subscribers ---
    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/pf/pose/odom", 1,
        std::bind(&RRT::pose_callback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 1,
        std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    // --- Allocate occupancy grid ---
    occupancy_grid_.assign(grid_width_ * grid_height_, 0);

    // --- Sampling distributions (car-local frame, ±lookahead_dist_) ---
    x_dist          = std::uniform_real_distribution<>(-lookahead_dist_, lookahead_dist_ * 2.0);
    y_dist          = std::uniform_real_distribution<>(-lookahead_dist_, lookahead_dist_);
    goal_bias_dist  = std::uniform_real_distribution<>(0.0, 1.0);

    // --- Load waypoints CSV ---
    std::string waypoints_file = this->get_parameter("waypoints_file").as_string();
    if (!waypoints_file.empty()) {
        std::ifstream file(waypoints_file);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream ss(line);
                std::string token;
                std::array<double, 4> wp = {0.0, 0.0, 0.0, 1.0};
                if (std::getline(ss, token, ',')) wp[0] = std::stod(token);
                if (std::getline(ss, token, ',')) wp[1] = std::stod(token);
                if (std::getline(ss, token, ',')) wp[2] = std::stod(token);
                if (std::getline(ss, token, ',')) wp[3] = std::stod(token);
                waypoints_.push_back(wp);
            }
            RCLCPP_INFO(get_logger(), "Loaded %zu waypoints from %s",
                        waypoints_.size(), waypoints_file.c_str());
        } else {
            RCLCPP_WARN(get_logger(), "Could not open waypoints file: %s", waypoints_file.c_str());
        }
    } else {
        RCLCPP_WARN(get_logger(), "No waypoints_file parameter set. Goal biasing disabled.");
    }

    RCLCPP_INFO(get_logger(), "RRT node started. use_rrt_star=%s",
                use_rrt_star_ ? "true" : "false");
}

// ============================================================
//  scan_callback  — build local occupancy grid
// ============================================================
void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // Clear the grid
    std::fill(occupancy_grid_.begin(), occupancy_grid_.end(), 0);

    float angle = scan_msg->angle_min;

    const float car_chassis_radius = 0.1;
    
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i, angle += scan_msg->angle_increment) {
        float r = scan_msg->ranges[i];

        // Ignore invalid readings and anything too close to the car (could be part of the chassis)
        if (!std::isfinite(r) || r < scan_msg->range_min || r > scan_msg->range_max || r < car_chassis_radius) continue;

        // Hit point in car frame (x forward, y left)
        double hx = r * std::cos(angle);
        double hy = r * std::sin(angle);

        int gx, gy;
        if (!world_to_grid(hx, hy, gx, gy)) continue;

        // Mark occupied and inflate
        for (int dx = -inflation_radius_; dx <= inflation_radius_; ++dx) {
            for (int dy = -inflation_radius_; dy <= inflation_radius_; ++dy) {
                int nx = gx + dx;
                int ny = gy + dy;
                if (grid_in_bounds(nx, ny)) {
                    occupancy_grid_[ny * grid_width_ + nx] = 1;
                }
            }
        }
    }

    // Publish occupancy grid for RViz debugging
    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.frame_id      = "base_link";
    grid_msg.header.stamp         = scan_msg->header.stamp;
    grid_msg.info.resolution      = static_cast<float>(grid_resolution_);
    grid_msg.info.width           = grid_width_;
    grid_msg.info.height          = grid_height_;
    // Origin: bottom-left corner of the grid in car frame
    // Car sits at the center, so offset by half the grid size
    grid_msg.info.origin.position.x = -(grid_width_  / 2) * grid_resolution_;
    grid_msg.info.origin.position.y = -(grid_height_ / 2) * grid_resolution_;
    grid_msg.info.origin.orientation.w = 1.0;
    // OccupancyGrid expects values 0–100 (0=free, 100=occupied, -1=unknown)
    grid_msg.data.resize(grid_width_ * grid_height_);
    for (int i = 0; i < grid_width_ * grid_height_; ++i) {
        grid_msg.data[i] = occupancy_grid_[i] ? 100 : 0;
    }
    grid_pub_->publish(grid_msg);
}

// ============================================================
//  pose_callback  — RRT main loop
// ============================================================
void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    // --- Extract current pose ---
    car_x_ = pose_msg->pose.pose.position.x;
    car_y_ = pose_msg->pose.pose.position.y;

    // Convert quaternion to yaw
    double qx = pose_msg->pose.pose.orientation.x;
    double qy = pose_msg->pose.pose.orientation.y;
    double qz = pose_msg->pose.pose.orientation.z;
    double qw = pose_msg->pose.pose.orientation.w;
    car_yaw_ = std::atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz));

    // Publish waypoints at ~1 Hz (assuming ~20 Hz pose callbacks)
    if (pose_tick_++ % 200 == 0) visualize_waypoints();

    // --- Find goal in local frame ---
    double gx_local, gy_local;
    if (!find_goal(gx_local, gy_local)) {
        // No waypoints: aim straight ahead
        gx_local = lookahead_dist_;
        gy_local = 0.0;
    }
    goal_local_x_ = gx_local;
    goal_local_y_ = gy_local;
    visualize_goal(goal_local_x_, goal_local_y_);

    if (!has_obstacle_in_corridor()) {
        // ── No obstacle: follow raceline at waypoint speed ─────────────────
        int closest_idx = 0;
        double min_dist = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(waypoints_.size()); ++i) {
            double dx = waypoints_[i][0] - car_x_;
            double dy = waypoints_[i][1] - car_y_;
            double d  = std::hypot(dx, dy);
            if (d < min_dist) { min_dist = d; closest_idx = i; }
        }

        double L2 = goal_local_x_ * goal_local_x_ + goal_local_y_ * goal_local_y_;
        double steering_angle = 0.0;
        if (L2 > 1e-6)
            steering_angle = std::atan2(2.0 * wheelbase_ * goal_local_y_, L2);
        const double max_steer = 0.4;
        steering_angle = std::max(-max_steer, std::min(max_steer, steering_angle));

        double speed = waypoints_[closest_idx][3] * waypoint_speed_scale_;

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp         = this->get_clock()->now();
        drive_msg.drive.speed          = static_cast<float>(speed);
        drive_msg.drive.steering_angle = static_cast<float>(steering_angle);
        drive_pub_->publish(drive_msg);

        std::vector<RRT_Node> empty_tree;
        visualize_tree(empty_tree);
        return;
    }

    // ── Obstacle detected: run RRT ─────────────────────────────────────────
    std::vector<RRT_Node> tree;
    RRT_Node root;
    root.x       = 0.0;
    root.y       = 0.0;
    root.cost    = 0.0;
    root.parent  = -1;
    root.is_root = true;
    tree.push_back(root);

    bool path_found = false;

    for (int i = 0; i < max_iterations_; ++i) {
        std::vector<double> sampled = sample();
        int nearest_idx             = nearest(tree, sampled);
        RRT_Node new_node           = steer(tree[nearest_idx], sampled);

        if (check_collision(tree[nearest_idx], new_node)) continue;

        // ── RRT (basic) ────────────────────────────────────────────────────
        if (!use_rrt_star_) {
            new_node.parent = nearest_idx;
            new_node.cost   = 0.0;  // unused in basic RRT
            tree.push_back(new_node);
        }
        // ── RRT* ───────────────────────────────────────────────────────────
        else {
            std::vector<int> neighbors = near(tree, new_node);

            // Choose best parent from neighborhood
            int    best_parent = nearest_idx;
            double best_cost   = cost(tree, tree[nearest_idx]) + line_cost(tree[nearest_idx], new_node);

            for (int idx : neighbors) {
                double c = cost(tree, tree[idx]) + line_cost(tree[idx], new_node);
                if (c < best_cost && !check_collision(tree[idx], new_node)) {
                    best_cost   = c;
                    best_parent = idx;
                }
            }

            new_node.parent = best_parent;
            new_node.cost   = best_cost;
            tree.push_back(new_node);

            int new_idx = static_cast<int>(tree.size()) - 1;

            // Rewire: update neighbors that benefit from routing through new_node
            for (int idx : neighbors) {
                if (idx == best_parent) continue;
                double candidate = cost(tree, new_node) + line_cost(new_node, tree[idx]);
                if (candidate < cost(tree, tree[idx]) && !check_collision(new_node, tree[idx])) {
                    tree[idx].parent = new_idx;
                    tree[idx].cost   = candidate;
                }
            }
        }
        // ──────────────────────────────────────────────────────────────────

        if (is_goal(tree.back(), goal_local_x_, goal_local_y_)) {
            std::vector<RRT_Node> path = find_path(tree, tree.back());
            path = shortcut_path(path);
            last_path_ = path;
            visualize_path(path);
            execute_path(path);
            path_found = true;
            break;
        }
    }

    visualize_tree(tree);

    if (!path_found) {
        if (!last_path_.empty()) visualize_path(last_path_);

        double L2 = goal_local_x_ * goal_local_x_ + goal_local_y_ * goal_local_y_;
        double fallback_steer = 0.0;
        if (L2 > 1e-6) {
            fallback_steer = std::atan2(2.0 * wheelbase_ * goal_local_y_, L2);
            const double max_steer = 0.4;
            fallback_steer = std::max(-max_steer, std::min(max_steer, fallback_steer));
        }
        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.header.stamp         = this->get_clock()->now();
        drive_msg.drive.speed          = 0.5;
        drive_msg.drive.steering_angle = static_cast<float>(fallback_steer);
        drive_pub_->publish(drive_msg);
    }
}

// ============================================================
//  sample  — return random point in local free space
// ============================================================
std::vector<double> RRT::sample() {
    std::vector<double> sampled_point(2);

    if (goal_bias_dist(gen) < goal_bias_prob_) {
        sampled_point[0] = goal_local_x_;
        sampled_point[1] = goal_local_y_;
    } else {
        // Sample in a box that always spans from the car (0,0) to the goal,
        // with padding equal to max_expansion_dist_ on each side.
        // This keeps samples focused on the relevant corridor regardless of
        // which direction (forward, sideways, or behind) the goal lies.
        double pad  = max_expansion_dist_ * 2.0;
        double xmin = std::min(0.0, goal_local_x_) - pad;
        double xmax = std::max(0.0, goal_local_x_) + pad;
        double ymin = std::min(0.0, goal_local_y_) - pad;
        double ymax = std::max(0.0, goal_local_y_) + pad;

        sampled_point[0] = std::uniform_real_distribution<>(xmin, xmax)(gen);
        sampled_point[1] = std::uniform_real_distribution<>(ymin, ymax)(gen);
    }

    return sampled_point;
}

// ============================================================
//  nearest  — index of closest node in tree to sampled_point
// ============================================================
int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    int nearest_node  = 0;
    double min_dist2  = std::numeric_limits<double>::max();

    for (int i = 0; i < static_cast<int>(tree.size()); ++i) {
        double dx    = tree[i].x - sampled_point[0];
        double dy    = tree[i].y - sampled_point[1];
        double dist2 = dx * dx + dy * dy;  // squared — avoids sqrt in hot loop
        if (dist2 < min_dist2) {
            min_dist2    = dist2;
            nearest_node = i;
        }
    }

    return nearest_node;
}

// ============================================================
//  steer  — move from nearest_node toward sampled_point by at most max_expansion_dist_
// ============================================================
RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    RRT_Node new_node;

    double dx   = sampled_point[0] - nearest_node.x;
    double dy   = sampled_point[1] - nearest_node.y;
    double dist = std::sqrt(dx * dx + dy * dy);

    if (dist <= max_expansion_dist_) {
        new_node.x = sampled_point[0];
        new_node.y = sampled_point[1];
    } else {
        double scale = max_expansion_dist_ / dist;
        new_node.x   = nearest_node.x + dx * scale;
        new_node.y   = nearest_node.y + dy * scale;
    }

    new_node.cost    = 0.0;
    new_node.parent  = -1;
    new_node.is_root = false;

    return new_node;
}

// ============================================================
//  check_collision  — Bresenham line walk between two nodes
// ============================================================
bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    int x0, y0, x1, y1;
    if (!world_to_grid(nearest_node.x, nearest_node.y, x0, y0)) return true;
    if (!world_to_grid(new_node.x, new_node.y, x1, y1))         return true;

    // Bresenham's line algorithm
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    int cx = x0;
    int cy = y0;

    while (true) {
        if (!grid_in_bounds(cx, cy) || grid_occupied(cx, cy)) return true;
        if (cx == x1 && cy == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; cx += sx; }
        if (e2 <  dx) { err += dx; cy += sy; }
    }

    return false;  // no collision
}

// ============================================================
//  is_goal  — check if node is within goal_threshold_ of goal
// ============================================================
bool RRT::is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y) {
    double dx = latest_added_node.x - goal_x;
    double dy = latest_added_node.y - goal_y;
    return std::sqrt(dx * dx + dy * dy) < goal_threshold_;
}

// ============================================================
//  find_path  — backtrack from goal node to root
// ============================================================
std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node) {
    std::vector<RRT_Node> found_path;

    // Find the index of latest_added_node in the tree
    int idx = static_cast<int>(tree.size()) - 1;

    while (idx >= 0) {
        found_path.push_back(tree[idx]);
        if (tree[idx].is_root) break;
        idx = tree[idx].parent;
    }

    std::reverse(found_path.begin(), found_path.end());
    return found_path;
}

// ============================================================
//  RRT* — cost
// ============================================================
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // Cost is maintained incrementally in node.cost, so just return it.
    // For the root (is_root == true), cost is 0.
    return node.cost;
}

// ============================================================
//  RRT* — line_cost
// ============================================================
double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    double dx = n2.x - n1.x;
    double dy = n2.y - n1.y;
    return std::hypot(dx, dy);
}

// ============================================================
//  RRT* — near  (nodes within search_radius_)
// ============================================================
std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    std::vector<int> neighborhood;

    for (int i = 0; i < static_cast<int>(tree.size()); ++i) {
        double dx   = tree[i].x - node.x;
        double dy   = tree[i].y - node.y;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist <= search_radius_) {
            neighborhood.push_back(i);
        }
    }

    return neighborhood;
}

// ============================================================
//  Grid helpers
// ============================================================

// Convert car-frame world coordinates to grid indices.
// Car sits at the center of the grid.
bool RRT::world_to_grid(double wx, double wy, int &gx, int &gy) {
    int half_w = grid_width_  / 2;
    int half_h = grid_height_ / 2;

    gx = static_cast<int>(std::round(wx / grid_resolution_)) + half_w;
    gy = static_cast<int>(std::round(wy / grid_resolution_)) + half_h;

    return grid_in_bounds(gx, gy);
}

bool RRT::grid_in_bounds(int gx, int gy) {
    return (gx >= 0 && gx < grid_width_ && gy >= 0 && gy < grid_height_);
}

bool RRT::grid_occupied(int gx, int gy) {
    return occupancy_grid_[gy * grid_width_ + gx] != 0;
}

// ============================================================
//  has_obstacle_in_corridor  — any occupied cell in the narrow
//  forward corridor: x in [0, lookahead_dist_], |y| < 0.3 m
// ============================================================
bool RRT::has_obstacle_in_corridor() {
    const double half_width = 0.3;
    int x_cells = static_cast<int>(lookahead_dist_ / grid_resolution_);
    int y_half  = static_cast<int>(half_width / grid_resolution_);
    int cx      = grid_width_  / 2;
    int cy      = grid_height_ / 2;

    for (int dx = 0; dx <= x_cells; ++dx) {
        for (int dy = -y_half; dy <= y_half; ++dy) {
            int gx = cx + dx;
            int gy = cy + dy;
            if (grid_in_bounds(gx, gy) && grid_occupied(gx, gy)) return true;
        }
    }
    return false;
}

// ============================================================
//  find_goal  — pick lookahead waypoint in car-local frame
// ============================================================
bool RRT::find_goal(double &goal_local_x, double &goal_local_y) {
    if (waypoints_.empty()) return false;

    int n = static_cast<int>(waypoints_.size());

    double cos_yaw = std::cos(car_yaw_);
    double sin_yaw = std::sin(car_yaw_);

    // 1. Find the closest waypoint to the car
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        double dx = waypoints_[i][0] - car_x_;
        double dy = waypoints_[i][1] - car_y_;
        double d  = std::hypot(dx, dy);
        if (d < min_dist) { min_dist = d; closest_idx = i; }
    }

    // Helper lambda: scan forward from closest_idx for the first waypoint at
    // least `look` meters away. Fills lx/ly (car-local) and out_idx. Returns
    // false only if every waypoint is closer than `look`.
    auto scan_for_goal = [&](double look, double &lx, double &ly, int &out_idx) -> bool {
        for (int k = 0; k < n; ++k) {
            int idx       = (closest_idx + k) % n;
            double dx_map = waypoints_[idx][0] - car_x_;
            double dy_map = waypoints_[idx][1] - car_y_;
            if (std::hypot(dx_map, dy_map) >= look) {
                lx      =  cos_yaw * dx_map + sin_yaw * dy_map;
                ly      = -sin_yaw * dx_map + cos_yaw * dy_map;
                out_idx = idx;
                return true;
            }
        }
        return false;
    };

    // 2. Find goal at normal lookahead distance
    double lx = 0.0, ly = 0.0;
    int    normal_idx = -1;
    if (scan_for_goal(lookahead_dist_, lx, ly, normal_idx)) {
        goal_local_x = lx;
        goal_local_y = ly;
        return true;
    }

    // Fallback: closest waypoint in local frame
    double dx_map = waypoints_[closest_idx][0] - car_x_;
    double dy_map = waypoints_[closest_idx][1] - car_y_;
    goal_local_x  =  cos_yaw * dx_map + sin_yaw * dy_map;
    goal_local_y  = -sin_yaw * dx_map + cos_yaw * dy_map;
    return true;
}

// ============================================================
//  visualize_tree  — publish LINE_LIST marker of all tree edges
// ============================================================
void RRT::visualize_tree(std::vector<RRT_Node> &tree) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id    = "base_link";
    marker.header.stamp       = this->get_clock()->now();
    marker.ns                 = "rrt_tree";
    marker.id                 = 0;
    marker.type               = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action             = visualization_msgs::msg::Marker::ADD;
    marker.scale.x            = 0.02;
    marker.color.r            = 0.0f;
    marker.color.g            = 0.8f;
    marker.color.b            = 0.0f;
    marker.color.a            = 0.6f;

    for (int i = 1; i < static_cast<int>(tree.size()); ++i) {
        int p = tree[i].parent;
        if (p < 0) continue;

        geometry_msgs::msg::Point pt1, pt2;
        pt1.x = tree[p].x; pt1.y = tree[p].y; pt1.z = 0.0;
        pt2.x = tree[i].x; pt2.y = tree[i].y; pt2.z = 0.0;
        marker.points.push_back(pt1);
        marker.points.push_back(pt2);
    }

    tree_pub_->publish(marker);
}

// ============================================================
//  visualize_path  — publish LINE_STRIP marker for found path
// ============================================================
void RRT::visualize_path(std::vector<RRT_Node> &path) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id    = "base_link";
    marker.header.stamp       = this->get_clock()->now();
    marker.ns                 = "rrt_path";
    marker.id                 = 1;
    marker.type               = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action             = visualization_msgs::msg::Marker::ADD;
    marker.scale.x            = 0.05;
    marker.color.r            = 1.0f;
    marker.color.g            = 0.0f;
    marker.color.b            = 0.0f;
    marker.color.a            = 1.0f;

    for (auto &node : path) {
        geometry_msgs::msg::Point pt;
        pt.x = node.x;
        pt.y = node.y;
        pt.z = 0.0;
        marker.points.push_back(pt);
    }

    path_pub_->publish(marker);
}

// ============================================================
//  execute_path  — Pure Pursuit on the RRT path
// ============================================================
void RRT::execute_path(std::vector<RRT_Node> &path) {
    if (path.size() < 2) return;

    // Upsample the path by linear interpolation every 0.05 m
    const double interp_step = 0.05;
    std::vector<std::array<double, 2>> dense_path;

    for (int i = 0; i + 1 < static_cast<int>(path.size()); ++i) {
        double x0 = path[i].x,   y0 = path[i].y;
        double x1 = path[i+1].x, y1 = path[i+1].y;
        double seg_len = std::hypot(x1 - x0, y1 - y0);
        int    steps   = std::max(1, static_cast<int>(std::ceil(seg_len / interp_step)));

        for (int s = 0; s < steps; ++s) {
            double t = static_cast<double>(s) / steps;
            dense_path.push_back({x0 + t * (x1 - x0), y0 + t * (y1 - y0)});
        }
    }
    dense_path.push_back({path.back().x, path.back().y});

    // Pick the lookahead point
    double lx = dense_path.back()[0];
    double ly = dense_path.back()[1];

    double acc_dist = 0.0;
    for (int i = 1; i < static_cast<int>(dense_path.size()); ++i) {
        double dx = dense_path[i][0] - dense_path[i-1][0];
        double dy = dense_path[i][1] - dense_path[i-1][1];
        acc_dist += std::hypot(dx, dy);
        if (acc_dist >= pursuit_lookahead_) {
            lx = dense_path[i][0];
            ly = dense_path[i][1];
            break;
        }
    }

    // Pure Pursuit steering angle
    // The lookahead point is already in car-local frame (x forward, y left)
    double L2 = lx * lx + ly * ly;
    double steering_angle = 0.0;
    if (L2 > 1e-6) {
        steering_angle = std::atan2(2.0 * wheelbase_ * ly, L2);
    }

    // Clamp steering
    const double max_steer = 0.4;  // ~23 deg
    steering_angle = std::max(-max_steer, std::min(max_steer, steering_angle));

    // Speed: slow down on sharp turns
    double speed = max_speed_ * (1.0 - 0.5 * std::abs(steering_angle) / max_steer);

    ackermann_msgs::msg::AckermannDriveStamped drive_msg;
    drive_msg.header.stamp        = this->get_clock()->now();
    drive_msg.drive.speed          = static_cast<float>(speed);
    drive_msg.drive.steering_angle = static_cast<float>(steering_angle);
    drive_pub_->publish(drive_msg);
}

// ============================================================
//  shortcut_path  — greedy line-of-sight path shortcutting
//  Removes zigzag intermediate nodes by skipping any node whose
//  direct connection to a later node is collision-free.
// ============================================================
std::vector<RRT_Node> RRT::shortcut_path(std::vector<RRT_Node> &path) {
    if (path.size() <= 2) return path;

    std::vector<RRT_Node> shortened;
    shortened.push_back(path.front());

    int i = 0;
    while (i < static_cast<int>(path.size()) - 1) {
        int best = i + 1;
        // Only try to skip to nodes within max_shortcut_dist_ —
        // prevents shortcuts from jumping across corners where the
        // blocking wall has scrolled out of the local occupancy grid
        for (int j = static_cast<int>(path.size()) - 1; j > i + 1; --j) {
            double dx   = path[j].x - path[i].x;
            double dy   = path[j].y - path[i].y;
            double dist = std::hypot(dx, dy);
            if (dist > max_shortcut_dist_) continue;
            if (!check_collision(path[i], path[j])) {
                best = j;
                break;
            }
        }
        shortened.push_back(path[best]);
        i = best;
    }

    return shortened;
}

// ============================================================
//  visualize_waypoints  — POINTS marker in map frame
// ============================================================
void RRT::visualize_waypoints() {
    if (waypoints_.empty()) return;

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp    = this->get_clock()->now();
    marker.ns              = "rrt_waypoints";
    marker.id              = 2;
    marker.type            = visualization_msgs::msg::Marker::POINTS;
    marker.action          = visualization_msgs::msg::Marker::ADD;
    marker.scale.x         = 0.15;
    marker.scale.y         = 0.15;
    marker.color.r         = 1.0f;
    marker.color.g         = 1.0f;
    marker.color.b         = 0.0f;
    marker.color.a         = 1.0f;

    for (auto &wp : waypoints_) {
        geometry_msgs::msg::Point pt;
        pt.x = wp[0];
        pt.y = wp[1];
        pt.z = 0.0;
        marker.points.push_back(pt);
    }

    waypoints_pub_->publish(marker);
}

// ============================================================
//  visualize_goal  — SPHERE marker in car-local frame
// ============================================================
void RRT::visualize_goal(double goal_local_x, double goal_local_y) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "base_link";
    marker.header.stamp    = this->get_clock()->now();
    marker.ns              = "rrt_goal";
    marker.id              = 3;
    marker.type            = visualization_msgs::msg::Marker::SPHERE;
    marker.action          = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = goal_local_x;
    marker.pose.position.y = goal_local_y;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x         = 0.3;
    marker.scale.y         = 0.3;
    marker.scale.z         = 0.3;
    marker.color.r         = 1.0f;
    marker.color.g         = 0.4f;
    marker.color.b         = 0.0f;
    marker.color.a         = 1.0f;

    goal_pub_->publish(marker);
}