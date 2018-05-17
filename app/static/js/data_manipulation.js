"use strict"

$(document).ready (function(){
    $("#success-alert").hide();
    $("#saveData").click(function showAlert() {
       $("#success-alert").fadeTo(2000, 500).slideUp(500, function(){
       $("#success-alert").slideUp(500);
        });
    });
 });


var data;
var data_categories;
var chosenCategory = "";
var previousIndexCategory = 0;

function changeEntityType(i) {
    let index1 = Number(i) + 1

    if (data[i][1] == "1"){
        data[i][1] = "0";
        data[i][2] = "0";
        $("#category-group > li:nth-child("+ index1 +")").removeClass("text-danger");
        $("li:nth-child("+ index1 +")").removeClass("text-danger");
    } else {
        data[i][1] = "1"
        if (i > 0 && (data[i-1][2].startsWith("B-" + chosenCategory) || data[i-1][2].startsWith("I-" +chosenCategory))) {
            data[i][2] = "I-" + chosenCategory;
        } else {
            data[i][2] = "B-" + chosenCategory;
        }
        $("li:nth-child("+ index1 +")").addClass("text-danger");
        $("#category-group > li:nth-child("+ index1 +")").addClass("text-danger");
    }

    $("#category-group > li:nth-child("+ index1 +")").text(data[i][2]);
}

function changeCategoryName(index) {
    chosenCategory = data_categories[index - 1][0];
    $("#categories > div > div:nth-child("+ previousIndexCategory +") > div").removeClass("text-success");
    $("#categories > div > div:nth-child("+ index +") > div").addClass("text-success");

    previousIndexCategory = index
}

function loadSentence(sentences, categories) {
    data = sentences;
    data_categories = categories;

    changeCategoryName(1)
}

function saveData() {
    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(data),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
        $("#success-alert").text(data.msg)
        }
    });

}

function retrain_model() {
    $("#loading").show();
    $("#content").hide();

    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(["retrain"]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
        $("#loading").hide();
        $("#content").show();
        $("#score_text").text(data.msg)

        }
    });

}

function test_model() {
    $("#loading").show();
    $("#content").hide();

    let scoreText = document.getElementById("score_text");

    function format_func(item, index){
        scoreText.innerHTML = scoreText.innerHTML + item + "<br>";
    }

    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(["only_test"]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
        $("#loading").hide();
        $("#content").show();
        Array.from(data.msg[0].split("], [")).forEach(format_func);

        }
    });

}
