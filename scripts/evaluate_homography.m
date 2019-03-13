H = [-5.86065426e-05,  7.20049423e-04, -7.91927511e-01; 
     -8.32948552e-04,  1.29477073e-04,  4.92479502e-01;
     -2.65306532e-05, -3.54739645e-03,  1.00000000e+00];
src_raw = [[693, 495],[1137, 496],[228, 492],[259, 376], [528, 377], [795, 382], [1054, 386], [939, 350],[655, 356], [361, 354]];
dest_raw = [[2,0],[2,1.5],[2, -1.5],[5, -3], [5,-1], [5,1], [5,3], [7, 3], [7,0], [7,-3]];
src_x = zeros(1, length(src_raw)/2);
src_y = zeros(1, length(src_raw)/2);
dest_x = zeros(1,length(dest_raw)/2);
dest_y = zeros(1,length(dest_raw)/2);

%create x and y vectors
for i = 1:length(src_raw)
    if mod(i, 2) == 1 %odd, first number from each pair
        src_x(i/2+0.5) = src_raw(i);
        dest_x(i/2+.5) = dest_raw(i)/3.281;
    else %even
        src_y(i/2) = src_raw(i);
        dest_y(i/2) = dest_raw(i)/3.281;
    end
end
[src_X, src_Y] = meshgrid(src_x, src_y);
[dest_X, dest_Y] = meshgrid(dest_x, dest_y);

%actual distances
dist_actual = zeros(1, length(dest_x));%((dest_X).^2+(dest_Y).^2).^0.5;%zeros(1, length(dest_x));
for i = 1:length(dest_x)
    dist_actual(i) = sqrt((dest_x(i)^2+dest_y(i)^2));
end

%estimated distances
dist_est = zeros(1, length(src_x));
for i = 1:length(src_x)
    v = H*[src_x(i) src_y(i) 1]'; %matrix multiply
    for j = 1:length(v) %normalize
        v(j) = v(j)/v(3);
    end

    dist_est(i) = sqrt(v(1)^2+v(2)^2); %grab distances
end

err = dist_est-dist_actual;
scatter(dist_actual, err, 75, "filled");
grid on;
title('Homography Matrix Distance Calculation Errors');
xlabel('Distance to Pixel (m)')
ylabel('Estimation Error (m)')




    