count = 0; figure(); itemax = 100000; C = 0.0001;
%aList = [1 10 100]; pList = [1 10 30 100];
aList = [100]; pList = [30];
for p = pList
    for a = aList
        count = count+1;
        tic;
        [w, lossRec] = stoSVM(X,Y,C,a,p,itemax);
        timeConsumed = toc;
        pret = w'*Xt;
        % epsilon = (-Yt).*pret+1;
        % hingeloss = max(0,epsilon);
        % hingelosssum = sum(hingeloss)/mt
        binloss = (sign(pret)~=Yt);
        binlosssum = sum(binloss);
        accuracy = (1-binlosssum/mt)*100;
        subplot(size(pList,2),size(aList,2),count); plot(lossRec); title(['p = ',num2str(p),'; a= ',num2str(a),'; Accuracy = ', num2str(accuracy)]);
        disp(['p = ',num2str(p),'; a= ',num2str(a),'; Accuracy = ', num2str(accuracy),'; Time Consumed:',num2str(timeConsumed)]);
    end
end
        