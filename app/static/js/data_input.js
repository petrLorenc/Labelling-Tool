"use strict"

$(document).ready (function(){
    $("#success-alert").hide();
 });

function showAlertAndRedirect(redirect, time){
    $("#success-alert").hide();
    $("#success-alert").fadeTo(2000, 1).slideUp(1000, function(){
        setTimeout(function () {
            $("#success-alert").slideDown(1000);
            if (redirect) {
                window.location.href = redirect;
            }
        }, time)

    });
}

function load_data() {
    $("#loading").show();
    $("#options").hide();

    var dataName = prompt("Enter a name of your data: (location: './app/static/data/)'",
                            "manually_labeled.csv");

    dataName  = "./app/static/data/" + dataName;

    var labels = document.getElementById('labels');
    var saved_model = document.getElementById('saved_model');
    var zero_marker = document.getElementById('marker');


    $.ajax({
      type: "POST",
      url: "/",
      data: JSON.stringify(["load_data", dataName, labels.checked, saved_model.checked, zero_marker.checked]),
      contentType: "application/json; charset=utf-8",
      dataType: "json",
      success: function(data) {
            $("#success-alert").text(data.msg, 2000);
            $("#loading").hide();
            $("#options").show();

            console.log(data);
            showAlertAndRedirect(data.redirect);
        },
      error: function(data) {
            $("#success-alert").text(data.responseJSON.msg, 10000);
            $("#loading").hide();
            $("#options").show();

            console.log(data);
            showAlertAndRedirect(null);
        }
    });
}
