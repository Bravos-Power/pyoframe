\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xe78df0a16d54df5
Minimize
  max_distance
Subject To
 con_only_one_closest[0,0]: - ONE + is_closest[0,0,0] + is_closest[0,0,1]
   + is_closest[0,0,2] = 0
 con_only_one_closest[0,1]: - ONE + is_closest[0,1,0] + is_closest[0,1,1]
   + is_closest[0,1,2] = 0
 con_only_one_closest[0,2]: - ONE + is_closest[0,2,0] + is_closest[0,2,1]
   + is_closest[0,2,2] = 0
 con_only_one_closest[0,3]: - ONE + is_closest[0,3,0] + is_closest[0,3,1]
   + is_closest[0,3,2] = 0
 con_only_one_closest[1,0]: - ONE + is_closest[1,0,0] + is_closest[1,0,1]
   + is_closest[1,0,2] = 0
 con_only_one_closest[1,1]: - ONE + is_closest[1,1,0] + is_closest[1,1,1]
   + is_closest[1,1,2] = 0
 con_only_one_closest[1,2]: - ONE + is_closest[1,2,0] + is_closest[1,2,1]
   + is_closest[1,2,2] = 0
 con_only_one_closest[1,3]: - ONE + is_closest[1,3,0] + is_closest[1,3,1]
   + is_closest[1,3,2] = 0
 con_only_one_closest[2,0]: - ONE + is_closest[2,0,0] + is_closest[2,0,1]
   + is_closest[2,0,2] = 0
 con_only_one_closest[2,1]: - ONE + is_closest[2,1,0] + is_closest[2,1,1]
   + is_closest[2,1,2] = 0
 con_only_one_closest[2,2]: - ONE + is_closest[2,2,0] + is_closest[2,2,1]
   + is_closest[2,2,2] = 0
 con_only_one_closest[2,3]: - ONE + is_closest[2,3,0] + is_closest[2,3,1]
   + is_closest[2,3,2] = 0
 con_only_one_closest[3,0]: - ONE + is_closest[3,0,0] + is_closest[3,0,1]
   + is_closest[3,0,2] = 0
 con_only_one_closest[3,1]: - ONE + is_closest[3,1,0] + is_closest[3,1,1]
   + is_closest[3,1,2] = 0
 con_only_one_closest[3,2]: - ONE + is_closest[3,2,0] + is_closest[3,2,1]
   + is_closest[3,2,2] = 0
 con_only_one_closest[3,3]: - ONE + is_closest[3,3,0] + is_closest[3,3,1]
   + is_closest[3,3,2] = 0
 con_dist_x[0,0]: facility_position[0,1] + dist_x[0,0] = 0
 con_dist_x[0,1]: facility_position[1,1] + dist_x[0,1] = 0
 con_dist_x[0,2]: facility_position[2,1] + dist_x[0,2] = 0
 con_dist_x[1,0]: - 0.3333333333333333 ONE + facility_position[0,1]
   + dist_x[1,0] = 0
 con_dist_x[1,1]: - 0.3333333333333333 ONE + facility_position[1,1]
   + dist_x[1,1] = 0
 con_dist_x[1,2]: - 0.3333333333333333 ONE + facility_position[2,1]
   + dist_x[1,2] = 0
 con_dist_x[2,0]: - 0.6666666666666666 ONE + facility_position[0,1]
   + dist_x[2,0] = 0
 con_dist_x[2,1]: - 0.6666666666666666 ONE + facility_position[1,1]
   + dist_x[2,1] = 0
 con_dist_x[2,2]: - 0.6666666666666666 ONE + facility_position[2,1]
   + dist_x[2,2] = 0
 con_dist_x[3,0]: - ONE + facility_position[0,1] + dist_x[3,0] = 0
 con_dist_x[3,1]: - ONE + facility_position[1,1] + dist_x[3,1] = 0
 con_dist_x[3,2]: - ONE + facility_position[2,1] + dist_x[3,2] = 0
 con_dist_y[0,0]: facility_position[0,2] + dist_y[0,0] = 0
 con_dist_y[0,1]: facility_position[1,2] + dist_y[0,1] = 0
 con_dist_y[0,2]: facility_position[2,2] + dist_y[0,2] = 0
 con_dist_y[1,0]: - 0.3333333333333333 ONE + facility_position[0,2]
   + dist_y[1,0] = 0
 con_dist_y[1,1]: - 0.3333333333333333 ONE + facility_position[1,2]
   + dist_y[1,1] = 0
 con_dist_y[1,2]: - 0.3333333333333333 ONE + facility_position[2,2]
   + dist_y[1,2] = 0
 con_dist_y[2,0]: - 0.6666666666666666 ONE + facility_position[0,2]
   + dist_y[2,0] = 0
 con_dist_y[2,1]: - 0.6666666666666666 ONE + facility_position[1,2]
   + dist_y[2,1] = 0
 con_dist_y[2,2]: - 0.6666666666666666 ONE + facility_position[2,2]
   + dist_y[2,2] = 0
 con_dist_y[3,0]: - ONE + facility_position[0,2] + dist_y[3,0] = 0
 con_dist_y[3,1]: - ONE + facility_position[1,2] + dist_y[3,1] = 0
 con_dist_y[3,2]: - ONE + facility_position[2,2] + dist_y[3,2] = 0
 con_max_distance[0,0,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,0,0] - dist[0,0,0] >= 0
 con_max_distance[0,0,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,0,1] - dist[0,0,1] >= 0
 con_max_distance[0,0,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,0,2] - dist[0,0,2] >= 0
 con_max_distance[0,1,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,1,0] - dist[0,1,0] >= 0
 con_max_distance[0,1,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,1,1] - dist[0,1,1] >= 0
 con_max_distance[0,1,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,1,2] - dist[0,1,2] >= 0
 con_max_distance[0,2,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,2,0] - dist[0,2,0] >= 0
 con_max_distance[0,2,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,2,1] - dist[0,2,1] >= 0
 con_max_distance[0,2,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,2,2] - dist[0,2,2] >= 0
 con_max_distance[0,3,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,3,0] - dist[0,3,0] >= 0
 con_max_distance[0,3,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,3,1] - dist[0,3,1] >= 0
 con_max_distance[0,3,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[0,3,2] - dist[0,3,2] >= 0
 con_max_distance[1,0,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,0,0] - dist[1,0,0] >= 0
 con_max_distance[1,0,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,0,1] - dist[1,0,1] >= 0
 con_max_distance[1,0,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,0,2] - dist[1,0,2] >= 0
 con_max_distance[1,1,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,1,0] - dist[1,1,0] >= 0
 con_max_distance[1,1,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,1,1] - dist[1,1,1] >= 0
 con_max_distance[1,1,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,1,2] - dist[1,1,2] >= 0
 con_max_distance[1,2,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,2,0] - dist[1,2,0] >= 0
 con_max_distance[1,2,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,2,1] - dist[1,2,1] >= 0
 con_max_distance[1,2,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,2,2] - dist[1,2,2] >= 0
 con_max_distance[1,3,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,3,0] - dist[1,3,0] >= 0
 con_max_distance[1,3,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,3,1] - dist[1,3,1] >= 0
 con_max_distance[1,3,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[1,3,2] - dist[1,3,2] >= 0
 con_max_distance[2,0,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,0,0] - dist[2,0,0] >= 0
 con_max_distance[2,0,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,0,1] - dist[2,0,1] >= 0
 con_max_distance[2,0,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,0,2] - dist[2,0,2] >= 0
 con_max_distance[2,1,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,1,0] - dist[2,1,0] >= 0
 con_max_distance[2,1,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,1,1] - dist[2,1,1] >= 0
 con_max_distance[2,1,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,1,2] - dist[2,1,2] >= 0
 con_max_distance[2,2,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,2,0] - dist[2,2,0] >= 0
 con_max_distance[2,2,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,2,1] - dist[2,2,1] >= 0
 con_max_distance[2,2,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,2,2] - dist[2,2,2] >= 0
 con_max_distance[2,3,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,3,0] - dist[2,3,0] >= 0
 con_max_distance[2,3,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,3,1] - dist[2,3,1] >= 0
 con_max_distance[2,3,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[2,3,2] - dist[2,3,2] >= 0
 con_max_distance[3,0,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,0,0] - dist[3,0,0] >= 0
 con_max_distance[3,0,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,0,1] - dist[3,0,1] >= 0
 con_max_distance[3,0,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,0,2] - dist[3,0,2] >= 0
 con_max_distance[3,1,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,1,0] - dist[3,1,0] >= 0
 con_max_distance[3,1,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,1,1] - dist[3,1,1] >= 0
 con_max_distance[3,1,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,1,2] - dist[3,1,2] >= 0
 con_max_distance[3,2,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,2,0] - dist[3,2,0] >= 0
 con_max_distance[3,2,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,2,1] - dist[3,2,1] >= 0
 con_max_distance[3,2,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,2,2] - dist[3,2,2] >= 0
 con_max_distance[3,3,0]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,3,0] - dist[3,3,0] >= 0
 con_max_distance[3,3,1]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,3,1] - dist[3,3,1] >= 0
 con_max_distance[3,3,2]: 2.828 ONE + max_distance
   - 2.828 is_closest[3,3,2] - dist[3,3,2] >= 0
 con_dist[0,0,0]: [ - dist_x[0,0] ^2 - dist_y[0,0] ^2 + dist[0,0,0] ^2 ]
   = 0
 con_dist[0,0,1]: [ - dist_x[0,1] ^2 - dist_y[0,1] ^2 + dist[0,0,1] ^2 ]
   = 0
 con_dist[0,0,2]: [ - dist_x[0,2] ^2 - dist_y[0,2] ^2 + dist[0,0,2] ^2 ]
   = 0
 con_dist[0,1,0]: [ - dist_x[0,0] ^2 - dist_y[1,0] ^2 + dist[0,1,0] ^2 ]
   = 0
 con_dist[0,1,1]: [ - dist_x[0,1] ^2 - dist_y[1,1] ^2 + dist[0,1,1] ^2 ]
   = 0
 con_dist[0,1,2]: [ - dist_x[0,2] ^2 - dist_y[1,2] ^2 + dist[0,1,2] ^2 ]
   = 0
 con_dist[0,2,0]: [ - dist_x[0,0] ^2 - dist_y[2,0] ^2 + dist[0,2,0] ^2 ]
   = 0
 con_dist[0,2,1]: [ - dist_x[0,1] ^2 - dist_y[2,1] ^2 + dist[0,2,1] ^2 ]
   = 0
 con_dist[0,2,2]: [ - dist_x[0,2] ^2 - dist_y[2,2] ^2 + dist[0,2,2] ^2 ]
   = 0
 con_dist[0,3,0]: [ - dist_x[0,0] ^2 - dist_y[3,0] ^2 + dist[0,3,0] ^2 ]
   = 0
 con_dist[0,3,1]: [ - dist_x[0,1] ^2 - dist_y[3,1] ^2 + dist[0,3,1] ^2 ]
   = 0
 con_dist[0,3,2]: [ - dist_x[0,2] ^2 - dist_y[3,2] ^2 + dist[0,3,2] ^2 ]
   = 0
 con_dist[1,0,0]: [ - dist_x[1,0] ^2 - dist_y[0,0] ^2 + dist[1,0,0] ^2 ]
   = 0
 con_dist[1,0,1]: [ - dist_x[1,1] ^2 - dist_y[0,1] ^2 + dist[1,0,1] ^2 ]
   = 0
 con_dist[1,0,2]: [ - dist_x[1,2] ^2 - dist_y[0,2] ^2 + dist[1,0,2] ^2 ]
   = 0
 con_dist[1,1,0]: [ - dist_x[1,0] ^2 - dist_y[1,0] ^2 + dist[1,1,0] ^2 ]
   = 0
 con_dist[1,1,1]: [ - dist_x[1,1] ^2 - dist_y[1,1] ^2 + dist[1,1,1] ^2 ]
   = 0
 con_dist[1,1,2]: [ - dist_x[1,2] ^2 - dist_y[1,2] ^2 + dist[1,1,2] ^2 ]
   = 0
 con_dist[1,2,0]: [ - dist_x[1,0] ^2 - dist_y[2,0] ^2 + dist[1,2,0] ^2 ]
   = 0
 con_dist[1,2,1]: [ - dist_x[1,1] ^2 - dist_y[2,1] ^2 + dist[1,2,1] ^2 ]
   = 0
 con_dist[1,2,2]: [ - dist_x[1,2] ^2 - dist_y[2,2] ^2 + dist[1,2,2] ^2 ]
   = 0
 con_dist[1,3,0]: [ - dist_x[1,0] ^2 - dist_y[3,0] ^2 + dist[1,3,0] ^2 ]
   = 0
 con_dist[1,3,1]: [ - dist_x[1,1] ^2 - dist_y[3,1] ^2 + dist[1,3,1] ^2 ]
   = 0
 con_dist[1,3,2]: [ - dist_x[1,2] ^2 - dist_y[3,2] ^2 + dist[1,3,2] ^2 ]
   = 0
 con_dist[2,0,0]: [ - dist_x[2,0] ^2 - dist_y[0,0] ^2 + dist[2,0,0] ^2 ]
   = 0
 con_dist[2,0,1]: [ - dist_x[2,1] ^2 - dist_y[0,1] ^2 + dist[2,0,1] ^2 ]
   = 0
 con_dist[2,0,2]: [ - dist_x[2,2] ^2 - dist_y[0,2] ^2 + dist[2,0,2] ^2 ]
   = 0
 con_dist[2,1,0]: [ - dist_x[2,0] ^2 - dist_y[1,0] ^2 + dist[2,1,0] ^2 ]
   = 0
 con_dist[2,1,1]: [ - dist_x[2,1] ^2 - dist_y[1,1] ^2 + dist[2,1,1] ^2 ]
   = 0
 con_dist[2,1,2]: [ - dist_x[2,2] ^2 - dist_y[1,2] ^2 + dist[2,1,2] ^2 ]
   = 0
 con_dist[2,2,0]: [ - dist_x[2,0] ^2 - dist_y[2,0] ^2 + dist[2,2,0] ^2 ]
   = 0
 con_dist[2,2,1]: [ - dist_x[2,1] ^2 - dist_y[2,1] ^2 + dist[2,2,1] ^2 ]
   = 0
 con_dist[2,2,2]: [ - dist_x[2,2] ^2 - dist_y[2,2] ^2 + dist[2,2,2] ^2 ]
   = 0
 con_dist[2,3,0]: [ - dist_x[2,0] ^2 - dist_y[3,0] ^2 + dist[2,3,0] ^2 ]
   = 0
 con_dist[2,3,1]: [ - dist_x[2,1] ^2 - dist_y[3,1] ^2 + dist[2,3,1] ^2 ]
   = 0
 con_dist[2,3,2]: [ - dist_x[2,2] ^2 - dist_y[3,2] ^2 + dist[2,3,2] ^2 ]
   = 0
 con_dist[3,0,0]: [ - dist_x[3,0] ^2 - dist_y[0,0] ^2 + dist[3,0,0] ^2 ]
   = 0
 con_dist[3,0,1]: [ - dist_x[3,1] ^2 - dist_y[0,1] ^2 + dist[3,0,1] ^2 ]
   = 0
 con_dist[3,0,2]: [ - dist_x[3,2] ^2 - dist_y[0,2] ^2 + dist[3,0,2] ^2 ]
   = 0
 con_dist[3,1,0]: [ - dist_x[3,0] ^2 - dist_y[1,0] ^2 + dist[3,1,0] ^2 ]
   = 0
 con_dist[3,1,1]: [ - dist_x[3,1] ^2 - dist_y[1,1] ^2 + dist[3,1,1] ^2 ]
   = 0
 con_dist[3,1,2]: [ - dist_x[3,2] ^2 - dist_y[1,2] ^2 + dist[3,1,2] ^2 ]
   = 0
 con_dist[3,2,0]: [ - dist_x[3,0] ^2 - dist_y[2,0] ^2 + dist[3,2,0] ^2 ]
   = 0
 con_dist[3,2,1]: [ - dist_x[3,1] ^2 - dist_y[2,1] ^2 + dist[3,2,1] ^2 ]
   = 0
 con_dist[3,2,2]: [ - dist_x[3,2] ^2 - dist_y[2,2] ^2 + dist[3,2,2] ^2 ]
   = 0
 con_dist[3,3,0]: [ - dist_x[3,0] ^2 - dist_y[3,0] ^2 + dist[3,3,0] ^2 ]
   = 0
 con_dist[3,3,1]: [ - dist_x[3,1] ^2 - dist_y[3,1] ^2 + dist[3,3,1] ^2 ]
   = 0
 con_dist[3,3,2]: [ - dist_x[3,2] ^2 - dist_y[3,2] ^2 + dist[3,3,2] ^2 ]
   = 0
Bounds
 ONE = 1
 facility_position[0,1] <= 1
 facility_position[0,2] <= 1
 facility_position[1,1] <= 1
 facility_position[1,2] <= 1
 facility_position[2,1] <= 1
 facility_position[2,2] <= 1
 dist_x[0,0] free
 dist_x[0,1] free
 dist_x[0,2] free
 dist_x[1,0] free
 dist_x[1,1] free
 dist_x[1,2] free
 dist_x[2,0] free
 dist_x[2,1] free
 dist_x[2,2] free
 dist_x[3,0] free
 dist_x[3,1] free
 dist_x[3,2] free
 dist_y[0,0] free
 dist_y[0,1] free
 dist_y[0,2] free
 dist_y[1,0] free
 dist_y[1,1] free
 dist_y[1,2] free
 dist_y[2,0] free
 dist_y[2,1] free
 dist_y[2,2] free
 dist_y[3,0] free
 dist_y[3,1] free
 dist_y[3,2] free
Binaries
 is_closest[0,0,0] is_closest[0,0,1] is_closest[0,0,2] is_closest[0,1,0]
 is_closest[0,1,1] is_closest[0,1,2] is_closest[0,2,0] is_closest[0,2,1]
 is_closest[0,2,2] is_closest[0,3,0] is_closest[0,3,1] is_closest[0,3,2]
 is_closest[1,0,0] is_closest[1,0,1] is_closest[1,0,2] is_closest[1,1,0]
 is_closest[1,1,1] is_closest[1,1,2] is_closest[1,2,0] is_closest[1,2,1]
 is_closest[1,2,2] is_closest[1,3,0] is_closest[1,3,1] is_closest[1,3,2]
 is_closest[2,0,0] is_closest[2,0,1] is_closest[2,0,2] is_closest[2,1,0]
 is_closest[2,1,1] is_closest[2,1,2] is_closest[2,2,0] is_closest[2,2,1]
 is_closest[2,2,2] is_closest[2,3,0] is_closest[2,3,1] is_closest[2,3,2]
 is_closest[3,0,0] is_closest[3,0,1] is_closest[3,0,2] is_closest[3,1,0]
 is_closest[3,1,1] is_closest[3,1,2] is_closest[3,2,0] is_closest[3,2,1]
 is_closest[3,2,2] is_closest[3,3,0] is_closest[3,3,1] is_closest[3,3,2]
End
