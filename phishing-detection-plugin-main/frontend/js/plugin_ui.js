var background = chrome.extension.getBackgroundPage();
var colors = {
    "-1":"#408964",
    "0":"#f2de38",
    "1":"#ff0000"
};
var featureList = document.getElementById("features");

chrome.tabs.query({ currentWindow: true, active: true }, function(tabs){
    var result = background.results[tabs[0].id];
    var isPhish = background.isPhish[tabs[0].id];
    var legitimatePercent = background.legitimatePercents[tabs[0].id];

    for(var key in result){
        var newFeature = document.createElement("li");
        //console.log(key);
        newFeature.textContent = key;
        //newFeature.className = "rounded";
        newFeature.style.backgroundColor=colors[result[key]];
        featureList.appendChild(newFeature);
    }
    
    $("#site_score").text(parseInt(legitimatePercent)+"%");
    if(isPhish) {
        $("#res-circle").css("background", "#ff0000");
        $("#site_msg").text("Warning!! You're being phished.");
        $("#site_score").text(parseInt(legitimatePercent)-20+"%");
    }
});

