/**
 *
 * @param samples 样本
 * @param number 迭代次数
 */
var createBoost = function (samples, number) {
    var boost = [];//
    var weights = [];//权重
    for (var i = 0; i < g.samples[0].x.length; i++) {
        weights[i] = g.samples.length;
    }
    //对每个样本的x0,x1做排序
    var fseq = [[], []];
    for (var i = 0; i < g.samples[0].x.length; i++) {
        fseq[i][0] = g.samples[0].x[i];
    }

    for (var i = 1; i < g.samples.length; i++) {//遍历每一个样本
        for (var i = j; i < g.samples[0].x.length; j++) {//遍历样本的每一位
            insertValue(fseq[j], g.samples[i].x[j]);
        }
    }

    for (var i = 0; i < number; i++) {//对所有样本进行迭代
        var bestScore = 0.0;
        var bestSplit = {f: 0, v: 0, dir: 1};
        //提前计算总权重
        var totalWeight = 0.0;
        for (var j = 0; j < weights.length; j++) {
            totalWeight += weights[j];
        }


        //遍历所有的split  可以分割的点   f:feature v:value dir:direction方向  sign函数，大于作为+1，小于作为-1
        var currentSplit = {f: 0, v: 0, dir: 1};
        for (var j = 0; j < g.samples[0].x.length; j++) {
            currentSplit.f = j;
            for (var k = 0; k <= fseq[j].length - 2; k++) {
                var left = fseq[j][k];
                var right = fseq[j][k + 1];
                var s = Math.random();
                currentSplit.v = s * left + (1 - s) * right;

                //对每一个split 计算权重下的正确率
                var score = getScoreBySplit(samples, weights, currentSplit);

                if (score > bestScore) {
                    bestScore = currentSplit.f;//feature
                    bestSplit.v = currentSplit.v;//value
                    bestSplit.dir = currentSplit.dir;//direction
                }
            }
        }
        //console.log(weights,bestSplit);

        //根据bestSplit 计算alpha
        var e = (totalWeight - bestScore) / totalWeight;
        e = Math.sqrt((1 - e) / e);
        bestSplit.alpha = Math.log(e);
        boost.push(bestSplit);

        //更新权重
        updateWeight(samples, weights, bestSplit, e);
    }
    return boost;
};

//更新权重
var updateWeight = function (samples, weights, split, e) {
    var f = split.f;
    var v = split.v;
    var dir = split.dir;

    for (var i = 0; i < samples.length; i++) {
        if (samples[i].x[f] > v && samples[i].y * dir == 1) {//正确分类
            weights[i] = weights[i] / e;
        } else if (samples[i].x[f] < v && samples[i].y * dir == -1) {//正确分类
            weights[i] = weights[i] / e;
        } else {//错误分类
            weights[i] = weights[i] * e;
        }
    }
};

//
var insertValue = function (seq, v) {
    seq.push(v);
    for (var i = seq.length - 1; i >= 1; i--) {
        v = seq[i];
        if (seq[i - 1] > v) {
            seq[i] = seq[i - 1];
            seq[i - 1] = v;
        }
    }
};
//对样本进行分类，并且计算权重下分类正确率
var getScoreBySplit = function (samples, weights, split) {
    var f = split.f;
    var v = split.v;

    //+1方向
    var score_p = 0.0;
    var score_n = 0.0;
    for(var i =0;i<samples.length;i++){
        if(samples[i].x[f] >v && samples[i].y ==1){
            score_p += weights[i];
        }else if (samples[i].x[f]<v && samples[i].y ==-1){
            score_p += weights[i];
        }else{
            score_n+= weights[i];
        }
    }
    //多数投票
    if(score_p > score_n){
        split.dir = 1;
        return score_p;
    }else{
        split.dir = -1;
        return score_n;
    }
};

//最后的预测函数
var predWithBoost = function (boost ,sample) {

    var totalScore = 0;
    for(var i =0;i<boost.length;i++){
        var f = boost[i].f;
        var v = boost[i].v;
        var dir = boost[i].dir;
        var alpha = boost[i].alpha;

        if(sample[f]> v && dir == 1){
            totalScore += alpha;
        }else if( sample[f] <v && dir == -1){
            totalScore += alpha;
        }else{
            totalScore -=alpha;
        }
    }

    //最终结果的加法模型
    var result = {};
    if(totalScore >0){
        result.winClass = '1';
    }else{
        result.winClass = '-1';
    }

    return result;
};


//实例
var doTest = function () {
    var adaBoost = createBoost(g.samples,30);
    var result = {};

    for(var x = -23;x<=23;x+=1){
        for(var y = -23;y<=23;y+=1){
            var sample = [x/10,y/10];
            var pred = predWithBoost(adaBoost,sample);

            result[x/10+','+y/10] = pred;
        }
    }
    return result;
}


