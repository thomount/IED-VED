I = [1,2,3,4;2,3,4,5;3,4,5,6;4,5,6,7];

A = zeros(4);   %变换矩阵A,也可以通过函数dctmtx(n)求得
for i = 0:7
    for j = 0:7
        if i == 0
            a = sqrt(1/8);
        else
            a = sqrt(2/8);
        end
        A(i+1,j+1) = a*cos((j+0.5)*pi*i/8);
    end
end
%A
D1 = dct2(I)
D2 = zeros(4,4);
D3 = zeros(4,4);
for i = 1: 4
    D2(i, :) = dct(I(i, :));
end
for i = 1: 4
    D3(:, i) = dct(D2(:, i)')';
end
D3
