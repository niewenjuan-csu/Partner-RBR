function snbpssm(path, eachprotein)
disp(eachprotein)
filename1 = strcat(path,'\',eachprotein,'\AF-',eachprotein,'.pdb');
filename3 = strcat(path,'\',eachprotein,'\',eachprotein,'.prf');
posall=[];
row=100000000;
fid1=fopen(filename1,'r');
for i=1:row
    pl=fgetl(fid1);
    if ((pl(1:3)=='END')) % if ((pl(1:3)=='END')|(pl(1:3)=='TER'))
        break;
    elseif(((pl(1:4)=='ATOM')&(pl(14)=='C')&(pl(15)=='A')&(pl(22)=='A')))  % CA-CA<8
        pos=[str2num(pl(31:38)) str2num(pl(39:46)) str2num(pl(47:54))];
        posall=[posall;pos];
    end
end
fclose('all');
m=size(posall);
i=1;
j=1;
for i=1:m
    for j=1:m
        dis(i,j)=sqrt(sum((posall(i,:)-posall(j,:)).^2));
    end
end
[row,col] = find(dis<8);

fid3 = fopen(filename3, 'r');
C=textscan(fid3,'%d %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %[^\n]','headerlines', 3);
pssm = [C{1,3},C{1,4},C{1,5},C{1,6},C{1,7},C{1,8},C{1,9},C{1,10},C{1,11},C{1,12},C{1,13},C{1,14},C{1,15},C{1,16},C{1,17},C{1,18},C{1,19},C{1,20},C{1,21},C{1,22}];
D=pssm(row,:);
k = find([true;diff(col(:))~=0;true]);
n=size(k);
i=1;
for i=1:n-1
    PSSM(i,:)=mean(D(k(i):k(i+1)-1,:));
end

[QQ QC]=sort(dis,2,'ascend');%QQ:排序距离值，QC位置，QP前五个距离值，QPP前五位置%
QP= QQ(:,1:5); %改‘%%’选出距离最近的前n个，1:5就是前5，改变数量就可改变前n个残基%
QPP=QC(:,1:5); %%
SNB_PSSM=[];
for i=1:length(PSSM)
    P=[];
    for j=1:5 %%
        p=PSSM(QPP(i,j),:);
        P=[P p];
    end
    SNB_PSSM=[SNB_PSSM;P];
end
xlswrite(strcat(path,'\',eachprotein,'\',eachprotein,'_snb.xlsx'),SNB_PSSM); %输出xls文件;
fclose('all');
% A1 = SNB_PSSM;
end
