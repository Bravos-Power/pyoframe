minimize

obj:

max_distance

s.t.

con_only_one_closest[0,0]: is_closest[0,0,0] + is_closest[0,0,1] + is_closest[0,0,2] = 1
con_only_one_closest[0,1]: is_closest[0,1,0] + is_closest[0,1,1] + is_closest[0,1,2] = 1
con_only_one_closest[0,2]: is_closest[0,2,0] + is_closest[0,2,1] + is_closest[0,2,2] = 1
con_only_one_closest[0,3]: is_closest[0,3,0] + is_closest[0,3,1] + is_closest[0,3,2] = 1
con_only_one_closest[1,0]: is_closest[1,0,0] + is_closest[1,0,1] + is_closest[1,0,2] = 1
con_only_one_closest[1,1]: is_closest[1,1,0] + is_closest[1,1,1] + is_closest[1,1,2] = 1
con_only_one_closest[1,2]: is_closest[1,2,0] + is_closest[1,2,1] + is_closest[1,2,2] = 1
con_only_one_closest[1,3]: is_closest[1,3,0] + is_closest[1,3,1] + is_closest[1,3,2] = 1
con_only_one_closest[2,0]: is_closest[2,0,0] + is_closest[2,0,1] + is_closest[2,0,2] = 1
con_only_one_closest[2,1]: is_closest[2,1,0] + is_closest[2,1,1] + is_closest[2,1,2] = 1
con_only_one_closest[2,2]: is_closest[2,2,0] + is_closest[2,2,1] + is_closest[2,2,2] = 1
con_only_one_closest[2,3]: is_closest[2,3,0] + is_closest[2,3,1] + is_closest[2,3,2] = 1
con_only_one_closest[3,0]: is_closest[3,0,0] + is_closest[3,0,1] + is_closest[3,0,2] = 1
con_only_one_closest[3,1]: is_closest[3,1,0] + is_closest[3,1,1] + is_closest[3,1,2] = 1
con_only_one_closest[3,2]: is_closest[3,2,0] + is_closest[3,2,1] + is_closest[3,2,2] = 1
con_only_one_closest[3,3]: is_closest[3,3,0] + is_closest[3,3,1] + is_closest[3,3,2] = 1
con_dist_x[0,0]: dist_x[0,0] + facility_position[0,1] = 0
con_dist_x[0,1]: dist_x[0,1] + facility_position[1,1] = 0
con_dist_x[0,2]: dist_x[0,2] + facility_position[2,1] = 0
con_dist_x[1,0]: dist_x[1,0] + facility_position[0,1] = 0.3333333333333333
con_dist_x[1,1]: dist_x[1,1] + facility_position[1,1] = 0.3333333333333333
con_dist_x[1,2]: dist_x[1,2] + facility_position[2,1] = 0.3333333333333333
con_dist_x[2,0]: dist_x[2,0] + facility_position[0,1] = 0.6666666666666666
con_dist_x[2,1]: dist_x[2,1] + facility_position[1,1] = 0.6666666666666666
con_dist_x[2,2]: dist_x[2,2] + facility_position[2,1] = 0.6666666666666666
con_dist_x[3,0]: dist_x[3,0] + facility_position[0,1] = 1
con_dist_x[3,1]: dist_x[3,1] + facility_position[1,1] = 1
con_dist_x[3,2]: dist_x[3,2] + facility_position[2,1] = 1
con_dist_y[0,0]: dist_y[0,0] + facility_position[0,2] = 0
con_dist_y[0,1]: dist_y[0,1] + facility_position[1,2] = 0
con_dist_y[0,2]: dist_y[0,2] + facility_position[2,2] = 0
con_dist_y[1,0]: dist_y[1,0] + facility_position[0,2] = 0.3333333333333333
con_dist_y[1,1]: dist_y[1,1] + facility_position[1,2] = 0.3333333333333333
con_dist_y[1,2]: dist_y[1,2] + facility_position[2,2] = 0.3333333333333333
con_dist_y[2,0]: dist_y[2,0] + facility_position[0,2] = 0.6666666666666666
con_dist_y[2,1]: dist_y[2,1] + facility_position[1,2] = 0.6666666666666666
con_dist_y[2,2]: dist_y[2,2] + facility_position[2,2] = 0.6666666666666666
con_dist_y[3,0]: dist_y[3,0] + facility_position[0,2] = 1
con_dist_y[3,1]: dist_y[3,1] + facility_position[1,2] = 1
con_dist_y[3,2]: dist_y[3,2] + facility_position[2,2] = 1
con_dist[0,0,0]: [ dist[0,0,0] * dist[0,0,0] - dist_x[0,0] * dist_x[0,0] - dist_y[0,0] * dist_y[0,0] ] = 0
con_dist[0,0,1]: [ dist[0,0,1] * dist[0,0,1] - dist_x[0,1] * dist_x[0,1] - dist_y[0,1] * dist_y[0,1] ] = 0
con_dist[0,0,2]: [ dist[0,0,2] * dist[0,0,2] - dist_x[0,2] * dist_x[0,2] - dist_y[0,2] * dist_y[0,2] ] = 0
con_dist[0,1,0]: [ dist[0,1,0] * dist[0,1,0] - dist_x[0,0] * dist_x[0,0] - dist_y[1,0] * dist_y[1,0] ] = 0
con_dist[0,1,1]: [ dist[0,1,1] * dist[0,1,1] - dist_x[0,1] * dist_x[0,1] - dist_y[1,1] * dist_y[1,1] ] = 0
con_dist[0,1,2]: [ dist[0,1,2] * dist[0,1,2] - dist_x[0,2] * dist_x[0,2] - dist_y[1,2] * dist_y[1,2] ] = 0
con_dist[0,2,0]: [ dist[0,2,0] * dist[0,2,0] - dist_x[0,0] * dist_x[0,0] - dist_y[2,0] * dist_y[2,0] ] = 0
con_dist[0,2,1]: [ dist[0,2,1] * dist[0,2,1] - dist_x[0,1] * dist_x[0,1] - dist_y[2,1] * dist_y[2,1] ] = 0
con_dist[0,2,2]: [ dist[0,2,2] * dist[0,2,2] - dist_x[0,2] * dist_x[0,2] - dist_y[2,2] * dist_y[2,2] ] = 0
con_dist[0,3,0]: [ dist[0,3,0] * dist[0,3,0] - dist_x[0,0] * dist_x[0,0] - dist_y[3,0] * dist_y[3,0] ] = 0
con_dist[0,3,1]: [ dist[0,3,1] * dist[0,3,1] - dist_x[0,1] * dist_x[0,1] - dist_y[3,1] * dist_y[3,1] ] = 0
con_dist[0,3,2]: [ dist[0,3,2] * dist[0,3,2] - dist_x[0,2] * dist_x[0,2] - dist_y[3,2] * dist_y[3,2] ] = 0
con_dist[1,0,0]: [ dist[1,0,0] * dist[1,0,0] - dist_x[1,0] * dist_x[1,0] - dist_y[0,0] * dist_y[0,0] ] = 0
con_dist[1,0,1]: [ dist[1,0,1] * dist[1,0,1] - dist_x[1,1] * dist_x[1,1] - dist_y[0,1] * dist_y[0,1] ] = 0
con_dist[1,0,2]: [ dist[1,0,2] * dist[1,0,2] - dist_x[1,2] * dist_x[1,2] - dist_y[0,2] * dist_y[0,2] ] = 0
con_dist[1,1,0]: [ dist[1,1,0] * dist[1,1,0] - dist_x[1,0] * dist_x[1,0] - dist_y[1,0] * dist_y[1,0] ] = 0
con_dist[1,1,1]: [ dist[1,1,1] * dist[1,1,1] - dist_x[1,1] * dist_x[1,1] - dist_y[1,1] * dist_y[1,1] ] = 0
con_dist[1,1,2]: [ dist[1,1,2] * dist[1,1,2] - dist_x[1,2] * dist_x[1,2] - dist_y[1,2] * dist_y[1,2] ] = 0
con_dist[1,2,0]: [ dist[1,2,0] * dist[1,2,0] - dist_x[1,0] * dist_x[1,0] - dist_y[2,0] * dist_y[2,0] ] = 0
con_dist[1,2,1]: [ dist[1,2,1] * dist[1,2,1] - dist_x[1,1] * dist_x[1,1] - dist_y[2,1] * dist_y[2,1] ] = 0
con_dist[1,2,2]: [ dist[1,2,2] * dist[1,2,2] - dist_x[1,2] * dist_x[1,2] - dist_y[2,2] * dist_y[2,2] ] = 0
con_dist[1,3,0]: [ dist[1,3,0] * dist[1,3,0] - dist_x[1,0] * dist_x[1,0] - dist_y[3,0] * dist_y[3,0] ] = 0
con_dist[1,3,1]: [ dist[1,3,1] * dist[1,3,1] - dist_x[1,1] * dist_x[1,1] - dist_y[3,1] * dist_y[3,1] ] = 0
con_dist[1,3,2]: [ dist[1,3,2] * dist[1,3,2] - dist_x[1,2] * dist_x[1,2] - dist_y[3,2] * dist_y[3,2] ] = 0
con_dist[2,0,0]: [ dist[2,0,0] * dist[2,0,0] - dist_x[2,0] * dist_x[2,0] - dist_y[0,0] * dist_y[0,0] ] = 0
con_dist[2,0,1]: [ dist[2,0,1] * dist[2,0,1] - dist_x[2,1] * dist_x[2,1] - dist_y[0,1] * dist_y[0,1] ] = 0
con_dist[2,0,2]: [ dist[2,0,2] * dist[2,0,2] - dist_x[2,2] * dist_x[2,2] - dist_y[0,2] * dist_y[0,2] ] = 0
con_dist[2,1,0]: [ dist[2,1,0] * dist[2,1,0] - dist_x[2,0] * dist_x[2,0] - dist_y[1,0] * dist_y[1,0] ] = 0
con_dist[2,1,1]: [ dist[2,1,1] * dist[2,1,1] - dist_x[2,1] * dist_x[2,1] - dist_y[1,1] * dist_y[1,1] ] = 0
con_dist[2,1,2]: [ dist[2,1,2] * dist[2,1,2] - dist_x[2,2] * dist_x[2,2] - dist_y[1,2] * dist_y[1,2] ] = 0
con_dist[2,2,0]: [ dist[2,2,0] * dist[2,2,0] - dist_x[2,0] * dist_x[2,0] - dist_y[2,0] * dist_y[2,0] ] = 0
con_dist[2,2,1]: [ dist[2,2,1] * dist[2,2,1] - dist_x[2,1] * dist_x[2,1] - dist_y[2,1] * dist_y[2,1] ] = 0
con_dist[2,2,2]: [ dist[2,2,2] * dist[2,2,2] - dist_x[2,2] * dist_x[2,2] - dist_y[2,2] * dist_y[2,2] ] = 0
con_dist[2,3,0]: [ dist[2,3,0] * dist[2,3,0] - dist_x[2,0] * dist_x[2,0] - dist_y[3,0] * dist_y[3,0] ] = 0
con_dist[2,3,1]: [ dist[2,3,1] * dist[2,3,1] - dist_x[2,1] * dist_x[2,1] - dist_y[3,1] * dist_y[3,1] ] = 0
con_dist[2,3,2]: [ dist[2,3,2] * dist[2,3,2] - dist_x[2,2] * dist_x[2,2] - dist_y[3,2] * dist_y[3,2] ] = 0
con_dist[3,0,0]: [ dist[3,0,0] * dist[3,0,0] - dist_x[3,0] * dist_x[3,0] - dist_y[0,0] * dist_y[0,0] ] = 0
con_dist[3,0,1]: [ dist[3,0,1] * dist[3,0,1] - dist_x[3,1] * dist_x[3,1] - dist_y[0,1] * dist_y[0,1] ] = 0
con_dist[3,0,2]: [ dist[3,0,2] * dist[3,0,2] - dist_x[3,2] * dist_x[3,2] - dist_y[0,2] * dist_y[0,2] ] = 0
con_dist[3,1,0]: [ dist[3,1,0] * dist[3,1,0] - dist_x[3,0] * dist_x[3,0] - dist_y[1,0] * dist_y[1,0] ] = 0
con_dist[3,1,1]: [ dist[3,1,1] * dist[3,1,1] - dist_x[3,1] * dist_x[3,1] - dist_y[1,1] * dist_y[1,1] ] = 0
con_dist[3,1,2]: [ dist[3,1,2] * dist[3,1,2] - dist_x[3,2] * dist_x[3,2] - dist_y[1,2] * dist_y[1,2] ] = 0
con_dist[3,2,0]: [ dist[3,2,0] * dist[3,2,0] - dist_x[3,0] * dist_x[3,0] - dist_y[2,0] * dist_y[2,0] ] = 0
con_dist[3,2,1]: [ dist[3,2,1] * dist[3,2,1] - dist_x[3,1] * dist_x[3,1] - dist_y[2,1] * dist_y[2,1] ] = 0
con_dist[3,2,2]: [ dist[3,2,2] * dist[3,2,2] - dist_x[3,2] * dist_x[3,2] - dist_y[2,2] * dist_y[2,2] ] = 0
con_dist[3,3,0]: [ dist[3,3,0] * dist[3,3,0] - dist_x[3,0] * dist_x[3,0] - dist_y[3,0] * dist_y[3,0] ] = 0
con_dist[3,3,1]: [ dist[3,3,1] * dist[3,3,1] - dist_x[3,1] * dist_x[3,1] - dist_y[3,1] * dist_y[3,1] ] = 0
con_dist[3,3,2]: [ dist[3,3,2] * dist[3,3,2] - dist_x[3,2] * dist_x[3,2] - dist_y[3,2] * dist_y[3,2] ] = 0
con_max_distance[0,0,0]: max_distance - dist[0,0,0] -2.828 is_closest[0,0,0] >= -2.828
con_max_distance[0,0,1]: max_distance - dist[0,0,1] -2.828 is_closest[0,0,1] >= -2.828
con_max_distance[0,0,2]: max_distance - dist[0,0,2] -2.828 is_closest[0,0,2] >= -2.828
con_max_distance[0,1,0]: max_distance - dist[0,1,0] -2.828 is_closest[0,1,0] >= -2.828
con_max_distance[0,1,1]: max_distance - dist[0,1,1] -2.828 is_closest[0,1,1] >= -2.828
con_max_distance[0,1,2]: max_distance - dist[0,1,2] -2.828 is_closest[0,1,2] >= -2.828
con_max_distance[0,2,0]: max_distance - dist[0,2,0] -2.828 is_closest[0,2,0] >= -2.828
con_max_distance[0,2,1]: max_distance - dist[0,2,1] -2.828 is_closest[0,2,1] >= -2.828
con_max_distance[0,2,2]: max_distance - dist[0,2,2] -2.828 is_closest[0,2,2] >= -2.828
con_max_distance[0,3,0]: max_distance - dist[0,3,0] -2.828 is_closest[0,3,0] >= -2.828
con_max_distance[0,3,1]: max_distance - dist[0,3,1] -2.828 is_closest[0,3,1] >= -2.828
con_max_distance[0,3,2]: max_distance - dist[0,3,2] -2.828 is_closest[0,3,2] >= -2.828
con_max_distance[1,0,0]: max_distance - dist[1,0,0] -2.828 is_closest[1,0,0] >= -2.828
con_max_distance[1,0,1]: max_distance - dist[1,0,1] -2.828 is_closest[1,0,1] >= -2.828
con_max_distance[1,0,2]: max_distance - dist[1,0,2] -2.828 is_closest[1,0,2] >= -2.828
con_max_distance[1,1,0]: max_distance - dist[1,1,0] -2.828 is_closest[1,1,0] >= -2.828
con_max_distance[1,1,1]: max_distance - dist[1,1,1] -2.828 is_closest[1,1,1] >= -2.828
con_max_distance[1,1,2]: max_distance - dist[1,1,2] -2.828 is_closest[1,1,2] >= -2.828
con_max_distance[1,2,0]: max_distance - dist[1,2,0] -2.828 is_closest[1,2,0] >= -2.828
con_max_distance[1,2,1]: max_distance - dist[1,2,1] -2.828 is_closest[1,2,1] >= -2.828
con_max_distance[1,2,2]: max_distance - dist[1,2,2] -2.828 is_closest[1,2,2] >= -2.828
con_max_distance[1,3,0]: max_distance - dist[1,3,0] -2.828 is_closest[1,3,0] >= -2.828
con_max_distance[1,3,1]: max_distance - dist[1,3,1] -2.828 is_closest[1,3,1] >= -2.828
con_max_distance[1,3,2]: max_distance - dist[1,3,2] -2.828 is_closest[1,3,2] >= -2.828
con_max_distance[2,0,0]: max_distance - dist[2,0,0] -2.828 is_closest[2,0,0] >= -2.828
con_max_distance[2,0,1]: max_distance - dist[2,0,1] -2.828 is_closest[2,0,1] >= -2.828
con_max_distance[2,0,2]: max_distance - dist[2,0,2] -2.828 is_closest[2,0,2] >= -2.828
con_max_distance[2,1,0]: max_distance - dist[2,1,0] -2.828 is_closest[2,1,0] >= -2.828
con_max_distance[2,1,1]: max_distance - dist[2,1,1] -2.828 is_closest[2,1,1] >= -2.828
con_max_distance[2,1,2]: max_distance - dist[2,1,2] -2.828 is_closest[2,1,2] >= -2.828
con_max_distance[2,2,0]: max_distance - dist[2,2,0] -2.828 is_closest[2,2,0] >= -2.828
con_max_distance[2,2,1]: max_distance - dist[2,2,1] -2.828 is_closest[2,2,1] >= -2.828
con_max_distance[2,2,2]: max_distance - dist[2,2,2] -2.828 is_closest[2,2,2] >= -2.828
con_max_distance[2,3,0]: max_distance - dist[2,3,0] -2.828 is_closest[2,3,0] >= -2.828
con_max_distance[2,3,1]: max_distance - dist[2,3,1] -2.828 is_closest[2,3,1] >= -2.828
con_max_distance[2,3,2]: max_distance - dist[2,3,2] -2.828 is_closest[2,3,2] >= -2.828
con_max_distance[3,0,0]: max_distance - dist[3,0,0] -2.828 is_closest[3,0,0] >= -2.828
con_max_distance[3,0,1]: max_distance - dist[3,0,1] -2.828 is_closest[3,0,1] >= -2.828
con_max_distance[3,0,2]: max_distance - dist[3,0,2] -2.828 is_closest[3,0,2] >= -2.828
con_max_distance[3,1,0]: max_distance - dist[3,1,0] -2.828 is_closest[3,1,0] >= -2.828
con_max_distance[3,1,1]: max_distance - dist[3,1,1] -2.828 is_closest[3,1,1] >= -2.828
con_max_distance[3,1,2]: max_distance - dist[3,1,2] -2.828 is_closest[3,1,2] >= -2.828
con_max_distance[3,2,0]: max_distance - dist[3,2,0] -2.828 is_closest[3,2,0] >= -2.828
con_max_distance[3,2,1]: max_distance - dist[3,2,1] -2.828 is_closest[3,2,1] >= -2.828
con_max_distance[3,2,2]: max_distance - dist[3,2,2] -2.828 is_closest[3,2,2] >= -2.828
con_max_distance[3,3,0]: max_distance - dist[3,3,0] -2.828 is_closest[3,3,0] >= -2.828
con_max_distance[3,3,1]: max_distance - dist[3,3,1] -2.828 is_closest[3,3,1] >= -2.828
con_max_distance[3,3,2]: max_distance - dist[3,3,2] -2.828 is_closest[3,3,2] >= -2.828


bounds

facility_position[0,1] <= 1
facility_position[0,2] <= 1
facility_position[1,1] <= 1
facility_position[1,2] <= 1
facility_position[2,1] <= 1
facility_position[2,2] <= 1
is_closest[0,0,0] <= 1
is_closest[0,0,1] <= 1
is_closest[0,0,2] <= 1
is_closest[0,1,0] <= 1
is_closest[0,1,1] <= 1
is_closest[0,1,2] <= 1
is_closest[0,2,0] <= 1
is_closest[0,2,1] <= 1
is_closest[0,2,2] <= 1
is_closest[0,3,0] <= 1
is_closest[0,3,1] <= 1
is_closest[0,3,2] <= 1
is_closest[1,0,0] <= 1
is_closest[1,0,1] <= 1
is_closest[1,0,2] <= 1
is_closest[1,1,0] <= 1
is_closest[1,1,1] <= 1
is_closest[1,1,2] <= 1
is_closest[1,2,0] <= 1
is_closest[1,2,1] <= 1
is_closest[1,2,2] <= 1
is_closest[1,3,0] <= 1
is_closest[1,3,1] <= 1
is_closest[1,3,2] <= 1
is_closest[2,0,0] <= 1
is_closest[2,0,1] <= 1
is_closest[2,0,2] <= 1
is_closest[2,1,0] <= 1
is_closest[2,1,1] <= 1
is_closest[2,1,2] <= 1
is_closest[2,2,0] <= 1
is_closest[2,2,1] <= 1
is_closest[2,2,2] <= 1
is_closest[2,3,0] <= 1
is_closest[2,3,1] <= 1
is_closest[2,3,2] <= 1
is_closest[3,0,0] <= 1
is_closest[3,0,1] <= 1
is_closest[3,0,2] <= 1
is_closest[3,1,0] <= 1
is_closest[3,1,1] <= 1
is_closest[3,1,2] <= 1
is_closest[3,2,0] <= 1
is_closest[3,2,1] <= 1
is_closest[3,2,2] <= 1
is_closest[3,3,0] <= 1
is_closest[3,3,1] <= 1
is_closest[3,3,2] <= 1
-inf <= dist_x[0,0]
-inf <= dist_x[0,1]
-inf <= dist_x[0,2]
-inf <= dist_x[1,0]
-inf <= dist_x[1,1]
-inf <= dist_x[1,2]
-inf <= dist_x[2,0]
-inf <= dist_x[2,1]
-inf <= dist_x[2,2]
-inf <= dist_x[3,0]
-inf <= dist_x[3,1]
-inf <= dist_x[3,2]
-inf <= dist_y[0,0]
-inf <= dist_y[0,1]
-inf <= dist_y[0,2]
-inf <= dist_y[1,0]
-inf <= dist_y[1,1]
-inf <= dist_y[1,2]
-inf <= dist_y[2,0]
-inf <= dist_y[2,1]
-inf <= dist_y[2,2]
-inf <= dist_y[3,0]
-inf <= dist_y[3,1]
-inf <= dist_y[3,2]


binary

is_closest[0,0,0]
is_closest[0,0,1]
is_closest[0,0,2]
is_closest[0,1,0]
is_closest[0,1,1]
is_closest[0,1,2]
is_closest[0,2,0]
is_closest[0,2,1]
is_closest[0,2,2]
is_closest[0,3,0]
is_closest[0,3,1]
is_closest[0,3,2]
is_closest[1,0,0]
is_closest[1,0,1]
is_closest[1,0,2]
is_closest[1,1,0]
is_closest[1,1,1]
is_closest[1,1,2]
is_closest[1,2,0]
is_closest[1,2,1]
is_closest[1,2,2]
is_closest[1,3,0]
is_closest[1,3,1]
is_closest[1,3,2]
is_closest[2,0,0]
is_closest[2,0,1]
is_closest[2,0,2]
is_closest[2,1,0]
is_closest[2,1,1]
is_closest[2,1,2]
is_closest[2,2,0]
is_closest[2,2,1]
is_closest[2,2,2]
is_closest[2,3,0]
is_closest[2,3,1]
is_closest[2,3,2]
is_closest[3,0,0]
is_closest[3,0,1]
is_closest[3,0,2]
is_closest[3,1,0]
is_closest[3,1,1]
is_closest[3,1,2]
is_closest[3,2,0]
is_closest[3,2,1]
is_closest[3,2,2]
is_closest[3,3,0]
is_closest[3,3,1]
is_closest[3,3,2]

end
