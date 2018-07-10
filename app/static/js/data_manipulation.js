"use strict"

$(document).ready (function(){
    $("#success-alert").hide();
 });

function showAlert(){
    $("#success-alert").hide();
    $("#success-alert").fadeTo(2000, 500).slideUp(500, function(){
    $("#success-alert").slideUp(500);
    });
}

function loadSentence(sentences, categories) {
    data = sentences;
    data_categories = categories;

    changeCategoryName(1)
}

var data;
var data_categories;
var chosenCategory = "";
var previousIndexCategory = 0;

function changeEntityType(i) {
    var index1 = Number(i) + 2

    if (data[i][1] === "1"){
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
    $("#category"+ previousIndexCategory).removeClass("chosen_category");
    $("#category"+ index).addClass("chosen_category");

    previousIndexCategory = index
}

function saveData() {
    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(data),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
        $("#success-alert").text(data.msg);
        showAlert();
        }
    });

}

function retrain_model() {
    $("#loading").show();
    $("#content").hide();

    var epochs = parseInt(prompt("Please enter number of epochs", "20"));

    var dataName = prompt("Enter a name of your data: (location: './app/static/data/)'",
                            "CAPC_tokenized.csv");

    dataName  = "./app/static/data/" + dataName;

    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(["retrain", epochs, dataName]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
        $("#loading").hide();
        $("#content").show();

        var ctx = document.getElementById("loss");
        var ctx2 = document.getElementById("trainTest");
        show_train_loss(ctx, data);
        show_train_test(ctx2, data);
        }
    });
}

function test_model() {
    $("#loading").show();
    $("#content").hide();

    function format_func(items){
        var scoreText = document.getElementById("score_text");
        var table = ""
        table = "<table><tbody>";

        table += "<td> class </td>" +
                 "<td> precision </td>" +
                 "<td> recall </td>" +
                 "<td> f1_score </td>" +
                 "<td> support </td>";

        for (var i = 0; i < items.length; i++) {
            table += "<tr>";

            /* Must not forget the $ sign */
            table += "<td>" + items[i].class.toString() + "</td>" +
                     "<td>" + items[i].precision.toString() + "</td>" +
                     "<td>" + items[i].recall.toString() + "</td>" +
                     "<td>" + items[i].f1_score.toString() + "</td>" +
                     "<td>" + items[i].support.toString() + "</td>";

            table += '</tr>';
        }

        table += "</tbody></table>";
        scoreText.innerHTML = table;
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
        format_func(data.msg);
        console.log(data.msg);
        }
    });
}

function save_model() {

    var modelName = prompt("Enter a name for a new model: (location: './app/static/data/models/')",
                            "actual_model.pth.tar");

    var modelPath  = "./app/static/data/models/" + modelName;

    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(["save_model", modelPath]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
            $("#success-alert").text(data.msg);
            showAlert();
        }
    });
}

function load_model() {

    var modelName = prompt("Enter a name of new model: (location: './app/static/data/models/)'",
                            "actual_model.pth.tar");

    var modelPath  = "./app/static/data/models/" + modelName;

    $.ajax({
      type: "POST",
      url: "/validate",
      data: JSON.stringify(["load_model", modelPath]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
            $("#success-alert").text(data.msg);
            showAlert();
        }
    });
}

function show_train_loss(ctx, data) {
    ctx.height = 500
    ctx.width = 500
    var myChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.msg.epochs,
            datasets: [{
                data: data.msg.train_loss,
                label: "Loss",
                borderColor: "#3e95cd",
                fill: false
              }
            ]
          },
          options: {
            title: {
              display: true,
              text: 'Train loss'
            }
          }
        });
}

function show_train_test(ctx, data) {
    ctx.height = 500
    ctx.width = 500
    var myChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: data.msg.epochs,
            datasets: [{
                data: data.msg.train_acc,
                label: "Train error",
                borderColor: "#3e95cd",
                fill: false
              }, {
                data: data.msg.test_acc,
                label: "Test error",
                borderColor: "#8e5ea2",
                fill: false
              }
            ]
          },
          options: {
            title: {
              display: true,
              text: 'Train vs Test error'
            }
          }
        });
}