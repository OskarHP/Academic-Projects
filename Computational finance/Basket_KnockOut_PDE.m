clear; close all;
%% Pricing a 2D basket option (Knock&Out Call)
T=1; S10=100; S20=70; K=190;
% Payoff = max(S1(T)+S2(T)-K;0) if barriers are not touched
L1=80; U1=120; L2=60; U2=110; % Barriers for the two assets
r=0.01; sigma1=0.4; sigma2=0.35; rho=-0.3; % Parameters

MT=50; M1=20; M2=20; % Time, x1, x2

% BCs since we have a Knock&Out option
x1min=log(L1/S10); x1max=log(U1/S10);
x2min=log(L2/S20); x2max=log(U2/S20);

% Grids
dt=T/MT;
x1=linspace(x1min,x1max,M1); dx1=x1(2)-x1(1); % Step in x1
x2=linspace(x2min,x2max,M2); dx2=x2(2)-x2(1); % Step in x2
[X1,X2]=meshgrid(x1,x2); 
X=[X1(:) X2(:)]; numpoints=M1*M2; % X is a (M1*M2 x 2)-matrix of points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            i+1
%  i-M2       i      i+M2
%            i-1 
% Boundary in all four directions in 2d
% W=[1:M2]
% S=[1:M2:M1*M2]
% N=[M2:M2:M1*M2]
% E=[(M1-1)*M2+1:M1*M2]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Boundary_index = unique([ 1:M2,1:M2:M1*M2,M2:M2:M1*M2,(M1-1)*M2+1:M1*M2 ]); 
% 76 points if we have grid 20x20 (4 corners)

%% Construction of matrix
bound_points=length(Boundary_index);
inside_points=numpoints-bound_points;

M=spalloc(numpoints,numpoints,9*inside_points+bound_points); % 9*I+B are the nonzero points
for i=1:numpoints
    if min(abs(i-Boundary_index)) == 0 % If i is in Boundary_index
        M(i,i)=1;
    else
        M(i,i)=-1/dt-r; % Derivative wrt time and -rV term
        % first order derivative wrt x1
        coeff=r-sigma1^2/2;
        M(i,i+M2)=coeff/(2*dx1);
        M(i,i-M2)=-coeff/(2*dx1);
        % first order derivative wrt x2
        coeff=r-sigma2^2/2;
        M(i,i+1)=coeff/(2*dx2);
        M(i,i-1)=-coeff/(2*dx2);
        % second order derivative wrt x1
        coeff=sigma1^2/2;
        M(i,i+M2)=M(i,i+M2)+coeff/dx1^2;
        M(i,i)=M(i,i)-2*coeff/dx1^2;
        M(i,i-M2)=M(i,i-M2)+coeff/dx1^2;
        % second order derivative wrt x2
        coeff=sigma2^2/2;
        M(i,i+1)=M(i,i+1)+coeff/dx2^2;
        M(i,i)=M(i,i)-2*coeff/dx2^2;
        M(i,i-1)=M(i,i-1)+coeff/dx2^2;
        % mixed second order derivative
        coeff=rho*sigma1*sigma2;
        M(i,i+M2+1)=coeff/(4*dx1*dx2);
        M(i,i-M2+1)=-coeff/(4*dx1*dx2);
        M(i,i+M2-1)=-coeff/(4*dx1*dx2);
        M(i,i-M2-1)=coeff/(4*dx1*dx2);
    end
end

%% Implicit Euler
V=max(S10*exp(X(:,1))+S20*exp(X(:,2))-K,0); % Call option payoff
V(Boundary_index)=0; % Knock-out call so every boundary has value 0
for j=MT:-1:1 % Backwards in time looping
    V=M\(-V/dt);
end
figure
surf( S10*exp(X1),S20*exp(X2),reshape(V,size(X1)) )
price=griddata(S10*exp(X1),S20*exp(X2),reshape(V,size(X1)),...
    S10,S20);
        




